"""
exp25 — 커스텀 초경량: tangent 벡터에서 판별력 AU쌍만 sparse 선택 + 로지스틱.
근거(감 아님): 실험으로 확인된 4사실 기반.
 - pairwise 공분산이 답(0.885) → 입력=공분산 tangent
 - 로지스틱이 딥러닝 다 이김 → 로지스틱 유지, 앞에 최소 gating만
 - 무거우면 과적합 → 파라미터 극소(gating 153 + logreg)
 - 특정 AU쌍(exp20)이 판별 → sparse가 그 쌍 자동선택
기준: baseline(L2 로지스틱 전체tangent) 넘으면 "쌍 선택 도움".
방법: L1 로지스틱(sparse), ElasticNet, L2(baseline) 비교 + 어떤 쌍 살아남나.
결과 CSV. 예상 2-3분.
"""
import numpy as np, warnings, openpyxl, csv
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
NA=len(AU); SEEDS=list(range(10))
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
covs=np.array([ (lambda X: ledoit_wolf(X)[0])(seq(s)) for s in subs])
y=np.array([CL[s] for s in subs])
ts=TangentSpace(metric='riemann'); T=ts.fit_transform(covs)  # (n, 153)
print(f'n={len(y)}, tangent dim={T.shape[1]}, MDD{int(y.sum())}/HC{int((y==0).sum())}',flush=True)

def evalclf(make,name):
    aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
        for tr,te in skf.split(T,y):
            clf=make(); clf.fit(StandardScaler().fit(T[tr]).transform(T[tr]),y[tr])
            pb[te]=clf.decision_function(StandardScaler().fit(T[tr]).transform(T[te]))
        aucs.append(roc_auc_score(y,pb))
    print(f'  {name:28s} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)
    return np.mean(aucs),np.std(aucs)

print('=== 커스텀: sparse AU-pair 선택 비교 (10seed) ===',flush=True)
r={}
r['L2_baseline']=evalclf(lambda:LogisticRegression(max_iter=2000,class_weight='balanced',C=1.0),'L2 로지스틱(baseline)')
r['L1_sparse']=evalclf(lambda:LogisticRegression(max_iter=5000,class_weight='balanced',penalty='l1',solver='liblinear',C=0.5),'L1 sparse(쌍 선택)')
r['L1_strong']=evalclf(lambda:LogisticRegression(max_iter=5000,class_weight='balanced',penalty='l1',solver='liblinear',C=0.2),'L1 강한선택(C=0.2)')
r['elastic']=evalclf(lambda:LogisticRegression(max_iter=5000,class_weight='balanced',penalty='elasticnet',solver='saga',l1_ratio=0.5,C=0.5),'ElasticNet')

# 어떤 AU 쌍이 살아남나 (L1, 전체 학습)
pairs=[(AU[i],AU[j]) for i in range(NA) for j in range(i,NA)]
clf=LogisticRegression(max_iter=5000,class_weight='balanced',penalty='l1',solver='liblinear',C=0.5)
clf.fit(StandardScaler().fit_transform(T),y)
coef=np.abs(clf.coef_[0]); nz=(coef>1e-6).sum()
top=np.argsort(coef)[-8:][::-1]
print(f'\nL1 선택된 쌍: {nz}/{len(pairs)}개',flush=True)
print('상위 판별 AU쌍:',flush=True)
for t in top:
    if coef[t]>1e-6: print(f'  {pairs[t][0]}×{pairs[t][1]} ({coef[t]:.2f})',flush=True)

with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/sparse_pair_result.csv','w') as f:
    f.write('model,AUC,std\n')
    for k,(a,s) in r.items(): f.write(f'{k},{a:.4f},{s:.4f}\n')
print('\nDONE (CSV saved)',flush=True)
