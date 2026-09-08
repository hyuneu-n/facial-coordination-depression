"""exp93 — within-session coordination 시간 동역학 최소검증 (D-Vlog).
윈도우별 공분산(coordination)의 시간 궤적: 불안정성/경직성/드리프트가 우울 구분? 정적 대비 추가?
"""
import numpy as np, warnings, csv, os
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
W=30; STR=15; d=20   # 윈도우 30스텝(≈30s), stride 15
def nf(V):
    T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
def logm_vec(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None)
    L=U@np.diag(np.log(ev))@U.T
    iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def win_logs(seq):
    # seq:(T,d) → 윈도우별 log-Euclidean tangent 벡터 시퀀스
    outs=[]
    for s in range(0,len(seq)-W+1,STR):
        seg=seq[s:s+W]
        if seg.std(0).min()<1e-8: seg=seg+np.random.randn(*seg.shape)*1e-6
        C,_=ledoit_wolf(seg); C=C+1e-3*np.eye(d)
        outs.append(logm_vec(C))
    return np.array(outs)  # (K, d*(d+1)/2)
# 로드
rows=list(csv.DictReader(open(R/'labels.csv')))
raw=[];Y=[]
for r in rows:
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:V=np.load(fv)
    except:continue
    if V.ndim!=2 or V.shape[0]<W+STR or V.shape[1]!=136:continue
    raw.append(nf(V));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
Y=np.array(Y)
pca=PCA(d).fit(np.vstack([raw[i][::5] for i in range(len(raw))]))
STAT=[];DYN=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Z=(Z-Z.mean(0))/(Z.std(0)+1e-6)
    seq=win_logs(Z)  # (K, P)
    if len(seq)<3:
        STAT.append(np.zeros(seq.shape[1] if seq.ndim>1 else d*(d+1)//2));DYN.append(np.zeros(5));continue
    STAT.append(seq.mean(0))  # 정적 = 평균 coordination (이번 세미나 표현)
    dist=np.linalg.norm(np.diff(seq,axis=0),axis=1)  # 연속 윈도우 간 log-Euclid 거리
    velocity=dist.mean()              # 불안정성(평균 변화)
    volatility=dist.std()             # 변동성
    drift=np.linalg.norm(seq[-1]-seq[0])  # 순 드리프트
    pathlen=dist.sum()                # 총 경로길이
    rigidity=1.0/(velocity+1e-6)      # 경직성(변화 적을수록↑)
    DYN.append([velocity,volatility,drift,pathlen,rigidity])
STAT=np.array(STAT);DYN=np.array(DYN)
# 개별 동역학 feature 그룹차이
names=['velocity','volatility','drift','pathlen','rigidity']
print('=== 동역학 feature 그룹차이 (MDD vs HC) ===',flush=True)
for k,nm in enumerate(names):
    a=DYN[Y==1,k];b=DYN[Y==0,k]
    try:u,p=mannwhitneyu(a,b,alternative='two-sided')
    except:p=1
    print(f'  {nm:10s} MDD={a.mean():.3f} HC={b.mean():.3f} p={p:.2e}',flush=True)
# CV AUC 비교: 정적 / 동역학 / 결합
def cv_auc(X):
    skf=StratifiedKFold(5,shuffle=True,random_state=0);aucs=[]
    for tr,te in skf.split(X,Y):
        sc=StandardScaler().fit(X[tr]);Xt=sc.transform(X[tr]);Xe=sc.transform(X[te])
        m=LogisticRegression(max_iter=2000,C=0.5).fit(Xt,Y[tr])
        aucs.append(roc_auc_score(Y[te],m.predict_proba(Xe)[:,1]))
    return np.mean(aucs),np.std(aucs)
print(f'\n=== 5-fold CV AUC (n={len(Y)}, MDD={Y.sum()}) ===',flush=True)
for nm,X in [('정적 coordination',STAT),('동역학(5feat)',DYN),('결합',np.hstack([STAT,DYN]))]:
    m,s=cv_auc(X);print(f'  {nm:20s} AUC={m:.3f} ± {s:.3f}',flush=True)
print('DONE',flush=True)
