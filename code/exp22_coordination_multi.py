"""
exp22 — 협응(coordination)을 5가지 방법으로 측정 → 다 우울 잡나 (방법론 탄탄함).
CMDC 앵커(Q3+Q7). 각 방법 AUC 10seed + permutation p.
1) 공분산 Riemannian (기준, 0.885)
2) 시간지연 상관 (협응 타이밍)
3) MdRQA 경직도 (DET/LAM/ENTR)
4) HMM 상태전이 (dwell/transition)
5) NMF 근육시너지 (시너지 개수/재구성)
"협응 붕괴가 여러 각도로 일관" = loss-of-complexity 원리 뒷받침.
"""
import numpy as np, warnings, openpyxl, sys
_LOG=open('/home/hyuneun/disk_b/🟡facial-prodrome/code/exp22_result.txt','w')
def _p(*a):
    print(*a); print(*a,file=_LOG); _LOG.flush()
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
SEEDS=list(range(10))
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
NA=len(AU)
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0]); iID,iMDD=hd.index('ID'),hd.index('MDD')
CL={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def c_q(subj,q):
    f=C/subj/f'Q{q}.csv'
    if not f.exists(): return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try: oi=h.index('success'); ai=[h.index(c) for c in AU]
    except: return None
    fe=[]
    with open(f) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(fe) if len(fe)>=10 else None
def seq(subj,qs=[3,7]):
    parts=[c_q(subj,q) for q in qs]
    if any(p is None for p in parts): return None
    return np.vstack(parts)
subs=[s for s in CL if seq(s) is not None]
SEQ={s:seq(s) for s in subs}; y=np.array([CL[s] for s in subs])
_p(f'CMDC 앵커 Q3+Q7: n={len(subs)}, MDD{int(y.sum())}/HC{int((y==0).sum())}\n',flush=True)

# ===== 협응 측정 5종 → 피험자별 feature =====
def f_cov(X):  # 공분산 (Riemannian용 raw matrix 반환은 따로)
    c,_=ledoit_wolf(X); return c
def f_lagcorr(X,lag=3):  # 시간지연 상관 (협응 타이밍): AU_i(t) vs AU_j(t+lag)
    if len(X)<=lag: return np.zeros(NA*NA)
    A=X[:-lag]; B=X[lag:]
    A=(A-A.mean(0))/(A.std(0)+1e-6); B=(B-B.mean(0))/(B.std(0)+1e-6)
    M=(A.T@B)/len(A)  # NA x NA 지연상관
    return M.flatten()
def f_rqa(X):  # MdRQA 경직도: 재귀행렬 기반 DET,LAM,RR,ENTR (multivariate)
    Xn=(X-X.mean(0))/(X.std(0)+1e-6)
    # 거리행렬 → recurrence (임계 = 중앙값의 0.3)
    from scipy.spatial.distance import pdist,squareform
    D=squareform(pdist(Xn)); thr=np.percentile(D,10); R=(D<thr).astype(int)
    N=len(R); RR=R.sum()/(N*N)
    # 대각선 구조(DET): 길이2+ 대각선에 속한 점 비율
    def diag_lines(M):
        lengths=[]
        for k in range(-N+1,N):
            d=np.diag(M,k); c=0
            for v in d:
                if v: c+=1
                else:
                    if c>=2: lengths.append(c)
                    c=0
            if c>=2: lengths.append(c)
        return lengths
    dl=diag_lines(R); vl=diag_lines(R.T)  # 수직은 대칭이라 근사
    tot=R.sum()+1e-9
    DET=sum(dl)/tot; LAM=sum(vl)/tot
    ent=0.0
    if dl:
        u,cnt=np.unique(dl,return_counts=True); p=cnt/cnt.sum(); ent=-(p*np.log(p+1e-12)).sum()
    return np.array([RR,DET,LAM,ent])
def f_nmf(X,k=4):  # 근육 시너지: NMF, 재구성오차 + H 활성 통계
    Xp=X-X.min(0); Xp=np.clip(Xp,0,None)
    try:
        m=NMF(n_components=k,init='nndsvda',max_iter=300,random_state=0)
        W=m.fit_transform(Xp); H=m.components_
        recon_err=m.reconstruction_err_
        act=W.mean(0)  # 시너지별 평균 활성
        sparse=(H<0.01).mean()  # 시너지 sparseness
        return np.concatenate([[recon_err,sparse],act])
    except: return np.zeros(2+k)

# ===== 평가 =====
def evaluate_feat(feat_fn,name,riemann=False):
    if riemann:
        mats=np.array([f_cov(SEQ[s]) for s in subs])
        aucs=[]
        for sd in SEEDS:
            skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
            for tr,te in skf.split(mats,y):
                clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
                clf.fit(mats[tr],y[tr]); pb[te]=clf.decision_function(mats[te])
            aucs.append(roc_auc_score(y,pb))
    else:
        X=np.array([feat_fn(SEQ[s]) for s in subs]); X=np.nan_to_num(X)
        aucs=[]
        for sd in SEEDS:
            skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
            for tr,te in skf.split(X,y):
                clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
                clf.fit(X[tr],y[tr]); pb[te]=clf.decision_function(X[te])
            aucs.append(roc_auc_score(y,pb))
    aucs=np.array(aucs)
    # permutation p
    rng=np.random.RandomState(0); null=[]
    Xp=(np.array([f_cov(SEQ[s]) for s in subs]) if riemann else np.nan_to_num(np.array([feat_fn(SEQ[s]) for s in subs])))
    for _ in range(40):
        yp=rng.permutation(y)
        skf=StratifiedKFold(5,shuffle=True,random_state=42); pb=np.zeros(len(y))
        for tr,te in skf.split(Xp,y):
            if riemann:
                clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced'))
            else:
                clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced'))
            clf.fit(Xp[tr],yp[tr]); pb[te]=clf.decision_function(Xp[te]) if hasattr(clf,'decision_function') else 0
        null.append(roc_auc_score(yp,pb))
    p=(np.sum(np.array(null)>=aucs.mean())+1)/(len(null)+1)
    _p(f'  {name:22s} AUC={aucs.mean():.3f}±{aucs.std():.3f}  perm_p={p:.3f} {"**" if p<0.05 else ""}',flush=True)
    return aucs.mean()

_p('=== 협응 5종 측정 (CMDC 앵커, 10seed) ===',flush=True)
evaluate_feat(None,'1) 공분산 Riemannian',riemann=True)
evaluate_feat(f_lagcorr,'2) 시간지연 상관')
evaluate_feat(f_rqa,'3) MdRQA 경직도')
evaluate_feat(f_nmf,'4) NMF 근육시너지')
