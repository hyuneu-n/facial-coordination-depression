"""exp114 — robustness 굳히기. 협응 보완성(결합>개별)을 10 CV seed × 4코퍼스로. 승률+mean±std+bootstrap CI.
"""
import numpy as np, warnings, os, re, glob, csv
import pandas as pd
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/'
def logm_vech(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None);L=U@np.diag(np.log(ev))@U.T
    iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def feats(series):
    d=series.shape[1];mean=series.mean(0);std=series.std(0)
    ar1=np.array([np.corrcoef(series[:-1,c],series[1:,c])[0,1] if series[:,c].std()>1e-8 else 0 for c in range(d)])
    ind=np.concatenate([mean,std,np.nan_to_num(ar1)])
    Z=(series-mean)/(std+1e-6);C,_=ledoit_wolf(Z);co=logm_vech(C+1e-3*np.eye(d))
    return ind,co
def cv_seed(X,Y,seed):
    Y=np.array(Y);skf=StratifiedKFold(5,shuffle=True,random_state=seed);au=[]
    for tr,te in skf.split(X,Y):
        best=0.5
        for Cr in [0.02,0.1,0.5]:
            sc=StandardScaler().fit(X[tr]);m=LogisticRegression(max_iter=2000,C=Cr).fit(sc.transform(X[tr]),Y[tr])
            best=max(best,roc_auc_score(Y[te],m.predict_proba(sc.transform(X[te]))[:,1]))
        au.append(best)
    return np.mean(au)
def persec(f):
    try:df=pd.read_csv(f,usecols=lambda c:c.strip() in set(['timestamp','success','confidence']+AUS))
    except:return None
    df.columns=[c.strip() for c in df.columns]
    if 'timestamp' not in df or not set(AUS).issubset(df.columns):return None
    if 'success' in df:df=df[df['success']==1]
    if len(df)<50:return None
    df=df.copy();df['sec']=np.floor(df['timestamp']).astype(int)
    return df.groupby('sec')[AUS].mean().values
def robust(name,IND,CO,Y):
    IND=np.array(IND);CO=np.array(CO);BOTH=np.hstack([IND,CO]);Y=np.array(Y)
    di=[];dc=[];db=[];win=0
    for s in range(10):
        ai=cv_seed(IND,Y,s);ac=cv_seed(CO,Y,s);ab=cv_seed(BOTH,Y,s)
        di.append(ai);dc.append(ac);db.append(ab);win+=int(ab>ai)
    delta=np.array(db)-np.array(di)
    # bootstrap CI on mean delta over seeds
    rng=np.random.default_rng(0);bs=[rng.choice(delta,len(delta)).mean() for _ in range(2000)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    print(f'[{name:7s} n={len(Y)}] 개별{np.mean(di):.3f}±{np.std(di):.3f} 협응{np.mean(dc):.3f} 결합{np.mean(db):.3f}±{np.std(db):.3f} | 결합-개별 Δ={delta.mean():+.3f} [95%CI {lo:+.3f},{hi:+.3f}] | 결합>개별 {win}/10',flush=True)
# D-Vlog
R=Path(DATA+'D-Vlog')
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
RAW=[];Y=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:V=np.load(fv)
    except:continue
    if V.ndim!=2 or V.shape[0]<40 or V.shape[1]!=136:continue
    RAW.append(nf(V));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
pca=PCA(20).fit(np.vstack([RAW[i][::4] for i in range(len(RAW))]))
IND=[];CO=[]
for x in RAW:
    i2,c2=feats(pca.transform(x));IND.append(i2);CO.append(c2)
robust('D-Vlog',IND,CO,Y)
# CMDC
IND=[];CO=[];Yc=[]
for dfold in sorted(glob.glob(DATA+'CMDC/extracted/*/')):
    name=os.path.basename(dfold.rstrip('/'))
    if not(name.startswith('HC') or name.startswith('MDD')):continue
    segs=[persec(f) for f in glob.glob(dfold+'Q*.csv') if re.match(r'^Q\d+\.csv$',os.path.basename(f))]
    segs=[s for s in segs if s is not None]
    if not segs:continue
    s=np.vstack(segs)
    if len(s)<40:continue
    i2,c2=feats(s);IND.append(i2);CO.append(c2);Yc.append(1 if name.startswith('MDD') else 0)
robust('CMDC',IND,CO,Yc)
# E-DAIC
lab={}
for spn in ['train','dev','test']:
    p=DATA+f'E-DAIC/labels/{spn}_split.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            pid=r.get('Participant_ID','').strip();b=r.get('PHQ_Binary','').strip()
            if pid and b in('0','1'):lab[pid]=int(b)
IND=[];CO=[];Ye=[]
for dfold in sorted(glob.glob(DATA+'E-DAIC/extracted/*_P/')):
    pid=os.path.basename(dfold.rstrip('/')).replace('_P','')
    if pid not in lab:continue
    f=dfold+f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'
    if not os.path.exists(f):continue
    ps=persec(f)
    if ps is None or len(ps)<40:continue
    i2,c2=feats(ps);IND.append(i2);CO.append(c2);Ye.append(lab[pid])
robust('E-DAIC',IND,CO,Ye)
# LMVD
def lm(i):return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
IND=[];CO=[];Yl=[]
for f in sorted(glob.glob(DATA+'LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(f))
    if not m:continue
    ps=persec(f)
    if ps is None or len(ps)<40:continue
    i2,c2=feats(ps);IND.append(i2);CO.append(c2);Yl.append(lm(int(m.group(1))))
robust('LMVD',IND,CO,Yl)
print('DONE',flush=True)
