"""
exp22b — 협응 다각도 측정 (경량, permutation 없이 AUC만 빠르게).
CMDC 앵커 Q3+Q7. 각 방법 → 우울 AUC 10seed. 결과 즉시 파일 flush.
"""
import numpy as np, warnings, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings('ignore')
LOG=open('/home/hyuneun/disk_b/🟡facial-prodrome/code/exp22b_result.txt','w')
def P(*a): print(*a); print(*a,file=LOG); LOG.flush()
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
def seq(subj):
    parts=[c_q(subj,q) for q in [3,7]]
    if any(p is None for p in parts): return None
    return np.vstack(parts)
subs=[s for s in CL if seq(s) is not None]
SEQ={s:seq(s) for s in subs}; y=np.array([CL[s] for s in subs])
P(f'CMDC 앵커 Q3+Q7: n={len(subs)}, MDD{int(y.sum())}/HC{int((y==0).sum())}\n')

def f_cov(X): c,_=ledoit_wolf(X); return c
def f_lagcorr(X,lag=3):
    if len(X)<=lag: return np.zeros(NA*NA)
    A=X[:-lag]; B=X[lag:]
    A=(A-A.mean(0))/(A.std(0)+1e-6); B=(B-B.mean(0))/(B.std(0)+1e-6)
    return ((A.T@B)/len(A)).flatten()
def f_rqa(X):
    Xn=(X-X.mean(0))/(X.std(0)+1e-6)
    D=squareform(pdist(Xn)); thr=np.percentile(D,10); R=(D<thr).astype(int); N=len(R)
    RR=R.sum()/(N*N)
    def diaglines(M):
        L=[]
        for k in range(-N+1,N):
            d=np.diag(M,k); c=0
            for v in d:
                if v: c+=1
                else:
                    if c>=2: L.append(c)
                    c=0
            if c>=2: L.append(c)
        return L
    dl=diaglines(R); tot=R.sum()+1e-9; DET=sum(dl)/tot
    ent=0.0
    if dl:
        u,cnt=np.unique(dl,return_counts=True); p=cnt/cnt.sum(); ent=-(p*np.log(p+1e-12)).sum()
    return np.array([RR,DET,ent,np.mean(dl) if dl else 0])
def f_nmf(X,k=4):
    Xp=np.clip(X-X.min(0),0,None)
    try:
        m=NMF(n_components=k,init='nndsvda',max_iter=200,random_state=0)
        W=m.fit_transform(Xp); H=m.components_
        return np.concatenate([[m.reconstruction_err_,(H<0.01).mean()],W.mean(0),W.std(0)])
    except: return np.zeros(2+2*k)

def auc_vec(featfn,name):
    X=np.nan_to_num(np.array([featfn(SEQ[s]) for s in subs])); aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
        for tr,te in skf.split(X,y):
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(X[tr],y[tr]); pb[te]=clf.decision_function(X[te])
        aucs.append(roc_auc_score(y,pb))
    P(f'  {name:22s} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}')
def auc_riemann(name):
    mats=np.array([f_cov(SEQ[s]) for s in subs]); aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
        for tr,te in skf.split(mats,y):
            clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(mats[tr],y[tr]); pb[te]=clf.decision_function(mats[te])
        aucs.append(roc_auc_score(y,pb))
    P(f'  {name:22s} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}')

P('=== 협응 다각도 (CMDC 앵커, 10seed AUC) ===')
auc_riemann('1) 공분산 Riemannian')
auc_vec(f_lagcorr,'2) 시간지연 상관')
auc_vec(f_rqa,'3) MdRQA 경직도')
auc_vec(f_nmf,'4) NMF 시너지')
P('\nDONE')
