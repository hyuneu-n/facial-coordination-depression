"""exp117 — 멀티모달 필수성. 얼굴만(협응+활동량) vs +얼굴음성연동(feature). 연동이 탐지에 추가되나?
D-Vlog·LMVD·E-DAIC. 10seed + bootstrap CI. 협응필수 논증의 크로스모달 버전.
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
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/';NSEG=5;NR=250
def logm_vech(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None);L=U@np.diag(np.log(ev))@U.T
    iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def face_feat(series):
    Z=(series-series.mean(0))/(series.std(0)+1e-6);C,_=ledoit_wolf(Z)
    return np.concatenate([logm_vech(C+1e-3*np.eye(series.shape[1])),series.mean(0)])
def rs(x,n=NR):
    x=np.asarray(x,float);idx=np.linspace(0,len(x)-1,n);return np.interp(idx,np.arange(len(x)),x)
def cross_feat(face_act,voice_act):
    if len(face_act)<NSEG*4 or len(voice_act)<NSEG*2:return None
    f=rs(face_act);v=rs(voice_act);segs=[]
    for s in range(NSEG):
        i0=s*NR//NSEG;i1=(s+1)*NR//NSEG;a=f[i0:i1];b=v[i0:i1]
        segs.append(abs(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.0)
    g=abs(np.corrcoef(f,v)[0,1]) if f.std()>1e-9 and v.std()>1e-9 else 0.0
    return segs+[g]
def robust(name,FACE,CROSS,Y):
    FACE=np.array(FACE);CROSS=np.array(CROSS);BOTH=np.hstack([FACE,CROSS]);Y=np.array(Y)
    def cv(X,seed):
        skf=StratifiedKFold(5,shuffle=True,random_state=seed);au=[]
        for tr,te in skf.split(X,Y):
            best=.5
            for Cr in [0.02,0.1,0.5]:
                sc=StandardScaler().fit(X[tr]);m=LogisticRegression(max_iter=2000,C=Cr).fit(sc.transform(X[tr]),Y[tr])
                best=max(best,roc_auc_score(Y[te],m.predict_proba(sc.transform(X[te]))[:,1]))
            au.append(best)
        return np.mean(au)
    fa=[];bo=[];win=0
    for s in range(10):
        af=cv(FACE,s);ab=cv(BOTH,s);fa.append(af);bo.append(ab);win+=int(ab>af)
    delta=np.array(bo)-np.array(fa)
    rng=np.random.default_rng(0);bs=[rng.choice(delta,len(delta)).mean() for _ in range(2000)];lo,hi=np.percentile(bs,[2.5,97.5])
    print(f'[{name:7s} n={len(Y)}] 얼굴만 {np.mean(fa):.3f} → +연동 {np.mean(bo):.3f} | Δ={delta.mean():+.3f} [95%CI {lo:+.3f},{hi:+.3f}] | 개선 {win}/10',flush=True)
# ===== D-Vlog =====
R=Path(DATA+'D-Vlog')
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
RAW=[];AUD=[];Y=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
    if not(fv.exists() and fa.exists()):continue
    try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
    except:continue
    if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136 or Au.ndim!=2:continue
    RAW.append(nf(V));AUD.append(Au);Y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
pca=PCA(20).fit(np.vstack([RAW[i][::4] for i in range(len(RAW))]))
FA=[];CR=[];YY=[]
for V,Au,y in zip(RAW,AUD,Y):
    Z=pca.transform(V)
    face_act=np.concatenate([[0],np.linalg.norm(np.diff(V,axis=0),axis=1)]);voice=np.linalg.norm(Au,axis=1)
    cf=cross_feat(face_act,voice)
    if cf is None:continue
    FA.append(face_feat(Z));CR.append(cf);YY.append(y)
robust('D-Vlog',FA,CR,YY)
# ===== E-DAIC =====
lab={}
for spn in ['train','dev','test']:
    p=DATA+f'E-DAIC/labels/{spn}_split.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            pid=r.get('Participant_ID','').strip();b=r.get('PHQ_Binary','').strip()
            if pid and b in('0','1'):lab[pid]=int(b)
FA=[];CR=[];YY=[]
for dfold in sorted(glob.glob(DATA+'E-DAIC/extracted/*_P/')):
    pid=os.path.basename(dfold.rstrip('/')).replace('_P','')
    if pid not in lab:continue
    vf=dfold+f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv';af=dfold+f'features/{pid}_OpenSMILE2.3.0_egemaps.csv'
    if not(os.path.exists(vf) and os.path.exists(af)):continue
    try:
        dv=pd.read_csv(vf,usecols=lambda c:c.strip() in set(['success']+AUS));dv.columns=[c.strip() for c in dv.columns]
        if 'success' in dv:dv=dv[dv['success']==1]
        AU=dv[AUS].values.astype(float)
        loud=pd.read_csv(af,sep=';')['Loudness_sma3'].values.astype(float)
    except:continue
    if len(AU)<40:continue
    face_act=np.concatenate([[0],np.linalg.norm(np.diff(AU,axis=0),axis=1)])
    cf=cross_feat(face_act,loud)
    if cf is None:continue
    FA.append(face_feat(AU));CR.append(cf);YY.append(lab[pid])
robust('E-DAIC',FA,CR,YY)
# ===== LMVD =====
def lm(i):return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
aud={}
for f in glob.glob(DATA+'LMVD/extracted/Audio_feature/*.npy'):
    m=re.search(r'(\d+)',os.path.basename(f))
    if m:aud[int(m.group(1))]=f
FA=[];CR=[];YY=[]
for vf in sorted(glob.glob(DATA+'LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(vf))
    if not m:continue
    i=int(m.group(1))
    if i not in aud:continue
    try:dv=pd.read_csv(vf,usecols=lambda c:c.strip() in set(['success']+AUS));dv.columns=[c.strip() for c in dv.columns]
    except:continue
    if not set(AUS).issubset(dv.columns):continue
    if 'success' in dv:dv=dv[dv['success']==1]
    AU=dv[AUS].values.astype(float)
    if len(AU)<40:continue
    try:V=np.load(aud[i])
    except:continue
    face_act=np.concatenate([[0],np.linalg.norm(np.diff(AU,axis=0),axis=1)]);voice=np.linalg.norm(V,axis=1)
    cf=cross_feat(face_act,voice)
    if cf is None:continue
    FA.append(face_feat(AU));CR.append(cf);YY.append(lm(i))
robust('LMVD',FA,CR,YY)
print('DONE',flush=True)
