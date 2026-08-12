"""
exp23 — HMM 상태전이 (temporal coordination). 근거: EEG microstate/HMM 우울판별 이식.
CMDC 앵커 Q3+Q7 AU 시퀀스 → 공유 Gaussian HMM(K states, unsupervised) →
참가자별 dwell time, transition entropy, state occupancy → 우울 AUC.
손절선 0.7. 해석: 우울군 neutral dwell↑, transition↓ (정신운동지연).
"""
import numpy as np, warnings, openpyxl
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
LOG=open('/home/hyuneun/disk_b/🟡facial-prodrome/code/exp23_result.txt','w')
def P(*a): print(*a); print(*a,file=LOG); LOG.flush()
SEEDS=list(range(10)); Ks=[4,6,8]
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

# 전체 프레임으로 공유 HMM 학습 (unsupervised, 라벨 안 씀) → 참가자별 상태통계
def hmm_features(K,seed):
    # 표준화 (전체 기준)
    allX=np.vstack([SEQ[s] for s in subs]); mu,sd=allX.mean(0),allX.std(0)+1e-6
    lengths=[len(SEQ[s]) for s in subs]
    Xcat=np.vstack([(SEQ[s]-mu)/sd for s in subs])
    hmm=GaussianHMM(n_components=K,covariance_type='diag',n_iter=50,random_state=seed)
    hmm.fit(Xcat,lengths)
    feats=[]
    for s in subs:
        Xs=(SEQ[s]-mu)/sd
        states=hmm.predict(Xs)
        occ=np.bincount(states,minlength=K)/len(states)  # 상태 점유(dwell)
        # transition 통계
        trans=np.zeros((K,K))
        for a,b in zip(states[:-1],states[1:]): trans[a,b]+=1
        rs=trans.sum(1,keepdims=True); rs[rs==0]=1; T=trans/rs
        self_trans=np.diag(T).mean()  # 자기전이(머무름) 평균
        n_switch=(np.diff(states)!=0).mean()  # 전이율
        # transition entropy
        p=trans.flatten()/(trans.sum()+1e-9); tent=-(p[p>0]*np.log(p[p>0])).sum()
        n_states_used=(occ>0.02).sum()
        feats.append(np.concatenate([occ,[self_trans,n_switch,tent,n_states_used]]))
    return np.array(feats)

P('=== HMM 상태전이 (CMDC 앵커, 10seed) ===')
best=0
for K in Ks:
    aucs=[]
    for sd in SEEDS:
        X=np.nan_to_num(hmm_features(K,sd))
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
        for tr,te in skf.split(X,y):
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(X[tr],y[tr]); pb[te]=clf.decision_function(X[te])
        aucs.append(roc_auc_score(y,pb))
    m=np.mean(aucs); best=max(best,m)
    P(f'  K={K} states: AUC={m:.3f}±{np.std(aucs):.3f}')
P(f'\n[최고] AUC={best:.3f} → {"채택(>0.7) ✅" if best>=0.7 else "미달, 공분산이 최선"}')
P('(비교: 공분산 0.885 / 시간지연 0.802)')
P('DONE')
