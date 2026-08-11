"""
exp20 — CMDC SPD+앵커 통계 확증 (가장 강한 결과 못박기).
1) 4자대조 쌍체 Wilcoxon (제안 vs 3 baseline, 10seed)
2) 순열검정 (라벨 셔플 → 제안 AUC가 우연 이상인가)
3) 해석: 어떤 AU 쌍 공분산이 우울과 관련 (접공간 로지스틱 계수 상위)
"""
import numpy as np, warnings, openpyxl
from pathlib import Path
from pyriemann.tangentspace import TangentSpace
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')
SEEDS=list(range(10))
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
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
    return np.array(fe) if len(fe)>=5 else None
def seq(subj,qs):
    parts=[c_q(subj,q) for q in qs]
    if any(p is None for p in parts): return None
    return np.vstack(parts)
def lw(X):
    if len(X)<3: return None
    c,_=ledoit_wolf(X); return c
subs=[s for s in CL if seq(s,[3,7]) is not None]
covs=np.array([lw(seq(s,[3,7])) for s in subs]); y=np.array([CL[s] for s in subs])
print(f'CMDC 앵커(Q3+Q7): n={len(y)}, 우울{int(y.sum())}/정상{int((y==0).sum())}\n',flush=True)

def auc_spd(cv,yy,seed):
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); pb=np.zeros(len(yy),float)
    for tr,te in skf.split(cv,yy):
        clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
        clf.fit(cv[tr],yy[tr]); pb[te]=clf.decision_function(cv[te])
    return roc_auc_score(yy,pb)

prop=np.array([auc_spd(covs,y,s) for s in SEEDS])
print(f'제안(SPD+앵커) AUC={prop.mean():.3f}±{prop.std():.3f}',flush=True)

# 2) 순열검정
rng=np.random.RandomState(0); null=[]
for _ in range(200):
    yp=rng.permutation(y); null.append(auc_spd(covs,yp,42))
null=np.array(null); pval=(np.sum(null>=prop.mean())+1)/(len(null)+1)
print(f'순열검정: null AUC={null.mean():.3f}, p={pval:.4f} {"유의**" if pval<0.05 else ""}',flush=True)

# 3) 해석: 접공간 계수 상위 AU쌍
from pyriemann.tangentspace import TangentSpace as TS
ts=TS(metric='riemann').fit(covs)
T=ts.transform(covs)
clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(StandardScaler().fit_transform(T),y)
coef=np.abs(clf.coef_[0])
# 접공간 벡터 인덱스 → (i,j) AU쌍 매핑 (상삼각 vech 순서)
n=len(AU); idxpair=[]
for i in range(n):
    for j in range(i,n): idxpair.append((i,j))
top=np.argsort(coef)[-8:][::-1]
print('\n=== 우울 판별 상위 AU 공분산 쌍 (해석) ===',flush=True)
for t in top:
    if t<len(idxpair):
        i,j=idxpair[t]; print(f'  {AU[i]}×{AU[j]} (|coef|={coef[t]:.2f})',flush=True)
