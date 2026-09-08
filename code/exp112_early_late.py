"""exp112 — 세션 앞→뒤 협응 일관성 (전조 프록시). 앞절반/뒤절반 협응 tangent의 유사도.
우울=경직 이론이면 앞뒤 더 비슷(stuck)→유사도↑. 그룹차이+AUC, 다코퍼스. 재현되면 흥미로운 각.
"""
import numpy as np, warnings, os, re, glob, csv
import pandas as pd
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/'
def logm_vech(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None);L=U@np.diag(np.log(ev))@U.T
    iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def coord(series):
    Z=(series-series.mean(0))/(series.std(0)+1e-6);C,_=ledoit_wolf(Z);return logm_vech(C+1e-3*np.eye(series.shape[1]))
def early_late_sim(series):
    h=len(series)//2
    e=coord(series[:h]);l=coord(series[h:])
    cos=float(np.dot(e,l)/(np.linalg.norm(e)*np.linalg.norm(l)+1e-9))  # 앞-뒤 유사도
    return cos
def persec(f):
    try:df=pd.read_csv(f,usecols=lambda c:c.strip() in set(['timestamp','success','confidence']+AUS))
    except:return None
    df.columns=[c.strip() for c in df.columns]
    if 'timestamp' not in df or not set(AUS).issubset(df.columns):return None
    if 'success' in df:df=df[df['success']==1]
    if len(df)<60:return None
    df=df.copy();df['sec']=np.floor(df['timestamp']).astype(int)
    return df.groupby('sec')[AUS].mean().values
def report(name,S,Y):
    Y=np.array(Y);a=np.array(S)
    mMDD=a[Y==1].mean();mHC=a[Y==0].mean()
    try:_,p=mannwhitneyu(a[Y==1],a[Y==0],alternative='two-sided')
    except:p=1
    try:auc=roc_auc_score(Y,a);auc=max(auc,1-auc)
    except:auc=0.5
    dr='MDD>HC(더 stuck)' if mMDD>mHC else 'MDD<HC'
    print(f'[{name:7s} n={len(Y)}] 앞뒤유사도 MDD={mMDD:.3f} HC={mHC:.3f} [{dr}] p={p:.2e} 단일AUC={auc:.3f}',flush=True)
    return dr,p
res={}
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
    if V.ndim!=2 or V.shape[0]<80 or V.shape[1]!=136:continue
    RAW.append(nf(V));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
pca=PCA(20).fit(np.vstack([RAW[i][::5] for i in range(len(RAW))]))
S=[early_late_sim(pca.transform(x)) for x in RAW]
res['D-Vlog']=report('D-Vlog',S,Y)
# CMDC
S=[];Y=[]
for dfold in sorted(glob.glob(DATA+'CMDC/extracted/*/')):
    name=os.path.basename(dfold.rstrip('/'))
    if not(name.startswith('HC') or name.startswith('MDD')):continue
    segs=[persec(f) for f in glob.glob(dfold+'Q*.csv') if re.match(r'^Q\d+\.csv$',os.path.basename(f))]
    segs=[s for s in segs if s is not None]
    if not segs:continue
    s=np.vstack(segs)
    if len(s)<60:continue
    S.append(early_late_sim(s));Y.append(1 if name.startswith('MDD') else 0)
res['CMDC']=report('CMDC',S,Y)
# LMVD
def lm(i):return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
S=[];Y=[]
for f in sorted(glob.glob(DATA+'LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(f))
    if not m:continue
    ps=persec(f)
    if ps is None or len(ps)<60:continue
    S.append(early_late_sim(ps));Y.append(lm(int(m.group(1))))
res['LMVD']=report('LMVD',S,Y)
print('\n===== 방향 일치 =====',flush=True)
for k,(dr,p) in res.items():print(f'  {k}: {dr} (p={p:.1e})',flush=True)
print('DONE',flush=True)
