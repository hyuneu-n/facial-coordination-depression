"""exp95 — 시간해상도 통일 재현. LMVD를 timestamp로 초단위 집계 → D-Vlog(초단위)와 공정 비교.
동일 W=30초 윈도우. 동역학 방향이 두 코퍼스서 일치하면 진짜 신호.
"""
import numpy as np, warnings, csv, os, re, glob
import pandas as pd
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
W=30; STR=15  # 30초 윈도우, 15초 stride (양 코퍼스 동일, 이제 둘 다 초단위)
def logm_vec(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None)
    L=U@np.diag(np.log(ev))@U.T;iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def win_logs(seq,d):
    outs=[]
    for s in range(0,len(seq)-W+1,STR):
        seg=seq[s:s+W]
        if seg.std(0).min()<1e-8: seg=seg+np.random.randn(*seg.shape)*1e-6
        C,_=ledoit_wolf(seg);C=C+1e-3*np.eye(d);outs.append(logm_vec(C))
    return np.array(outs)
def spectral_entropy(x):
    x=x-x.mean()
    if x.std()<1e-9 or len(x)<4: return 0.0
    ps=np.abs(np.fft.rfft(x))**2;ps=ps[1:]
    if ps.sum()<1e-12: return 0.0
    p=ps/ps.sum();p=p[p>0];return float(-(p*np.log(p)).sum()/np.log(len(p)+1e-9))
def dynamics(seq):
    if len(seq)<4: return None,None
    mean=seq.mean(0);d2m=np.linalg.norm(seq-mean,axis=1);dstep=np.linalg.norm(np.diff(seq,axis=0),axis=1)
    velocity=dstep.mean();innovation=dstep.std()
    a,b=d2m[:-1],d2m[1:];ar1=np.corrcoef(a,b)[0,1] if a.std()>1e-9 and b.std()>1e-9 else 0.0
    return seq.mean(0),[velocity,ar1,innovation,spectral_entropy(d2m)]
def cv_auc(X,Y):
    skf=StratifiedKFold(5,shuffle=True,random_state=0);aucs=[]
    for tr,te in skf.split(X,Y):
        sc=StandardScaler().fit(X[tr]);m=LogisticRegression(max_iter=2000,C=0.5).fit(sc.transform(X[tr]),Y[tr])
        aucs.append(roc_auc_score(Y[te],m.predict_proba(sc.transform(X[te]))[:,1]))
    return np.mean(aucs),np.std(aucs)
NAMES=['velocity','AR1(경직)','innovation','complexity']
def report(corp,STAT,DYN,Y):
    STAT=np.array(STAT);DYN=np.array(DYN);Y=np.array(Y)
    print(f'\n########## {corp} (n={len(Y)}, MDD={int(Y.sum())}) ##########',flush=True)
    dirs={}
    for k,nm in enumerate(NAMES):
        a=DYN[Y==1,k];b=DYN[Y==0,k]
        try:u,p=mannwhitneyu(a,b,alternative='two-sided')
        except:p=1
        dr='MDD<HC' if a.mean()<b.mean() else 'MDD>HC';dirs[nm]=dr
        print(f'  {nm:12s} MDD={a.mean():.3f} HC={b.mean():.3f} [{dr}] p={p:.2e}',flush=True)
    for nm,X in [('정적',STAT),('동역학',DYN),('결합',np.hstack([STAT,DYN]))]:
        m,s=cv_auc(X,Y);print(f'  AUC {nm:8s}= {m:.3f} ± {s:.3f}',flush=True)
    return dirs
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
# ===== D-Vlog (초단위 landmark PCA20) =====
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
def nf(V):
    T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
raw=[];Yd=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:V=np.load(fv)
    except:continue
    if V.ndim!=2 or V.shape[0]<W+2*STR or V.shape[1]!=136:continue
    raw.append(nf(V));Yd.append(1 if r['label'].strip().lower().startswith('depress') else 0)
pca=PCA(20).fit(np.vstack([raw[i][::5] for i in range(len(raw))]))
S1=[];D1=[];Y1=[]
for i in range(len(Yd)):
    Z=pca.transform(raw[i]);Z=(Z-Z.mean(0))/(Z.std(0)+1e-6)
    st,dy=dynamics(win_logs(Z,20))
    if dy is None:continue
    S1.append(st);D1.append(dy);Y1.append(Yd[i])
dg=report('D-Vlog',S1,D1,Y1)
# ===== LMVD (timestamp → 초단위 집계) =====
def lab(i): return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
want=set(['timestamp','success','confidence']+AUS)
S2=[];D2=[];Y2=[]
for f in sorted(glob.glob('/home/hyuneun/disk_b/🟡facial-prodrome/data/LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(f))
    if not m:continue
    i=int(m.group(1))
    try:df=pd.read_csv(f,usecols=lambda c:c.strip() in want)
    except:continue
    df.columns=[c.strip() for c in df.columns]
    if 'timestamp' not in df or not set(AUS).issubset(df.columns):continue
    if 'success' in df:df=df[df['success']==1]
    if 'confidence' in df:df=df[df['confidence']>0.9]
    if len(df)<50:continue
    df['sec']=np.floor(df['timestamp']).astype(int)
    persec=df.groupby('sec')[AUS].mean().values  # 초단위 (T_sec x 17)
    if len(persec)<W+2*STR:continue
    Xs=(persec-persec.mean(0))/(persec.std(0)+1e-6)
    st,dy=dynamics(win_logs(Xs,17))
    if dy is None:continue
    S2.append(st);D2.append(dy);Y2.append(lab(i))
lg=report('LMVD',S2,D2,Y2) if len(set(Y2))>1 else {}
# ===== 방향 일치 판정 =====
print('\n===== 방향 일치 판정 =====',flush=True)
for nm in NAMES:
    ok='✅ 일치' if dg.get(nm)==lg.get(nm) else '❌ 반대'
    print(f'  {nm:12s} D-Vlog={dg.get(nm)} | LMVD={lg.get(nm)}  {ok}',flush=True)
print('DONE',flush=True)
