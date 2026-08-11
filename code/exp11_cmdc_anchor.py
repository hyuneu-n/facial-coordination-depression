"""
exp11 — CMDC 앵커 결합. 증상질문(Q3수면/Q7피로/Q1식욕) 조합 vs 전체 vs 최고단일.
지표 CCC/MAE/AUC. seed 편차도 출력(안정성 확인).
"""
import numpy as np, warnings, openpyxl
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
warnings.filterwarnings('ignore')
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU_R=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r',
      'AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
SEEDS=[42,1,2,3,4,5,6,7]
def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean();vy,vp=y.var(),yp.var();cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0])
iID,iMDD=hd.index('ID'),hd.index('MDD'); iP=[hd.index(f'PHQ-{i}') for i in range(1,10)]
lab={}
for r in rows[1:]:
    if r[iID] is None: continue
    try: tot=sum(int(r[i]) for i in iP if r[i] is not None)
    except: continue
    lab[str(r[iID]).strip()]=(tot,int(r[iMDD]))
def au_q(subj,q):
    f=C/subj/f'Q{q}.csv'
    if not f.exists(): return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try: oi=h.index('success'); ai=[h.index(c) for c in AU_R]
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
def statfeat(au): return np.concatenate([au.mean(0),au.std(0)])

def feat_qs(subj,qs):
    parts=[]
    for q in qs:
        a=au_q(subj,q)
        if a is None: return None
        parts.append(statfeat(a))
    return np.concatenate(parts)

def ev(qs,name):
    subs=[s for s in lab if feat_qs(s,qs) is not None]
    X=np.array([feat_qs(s,qs) for s in subs])
    ys=np.array([lab[s][0] for s in subs]); yb=np.array([lab[s][1] for s in subs])
    cccs,maes,aucs=[],[],[]
    for sd in SEEDS:
        kf=KFold(5,shuffle=True,random_state=sd); pr=np.zeros(len(ys))
        for tr,te in kf.split(X):
            sc=StandardScaler().fit(X[tr]); m=Ridge(alpha=10).fit(sc.transform(X[tr]),ys[tr]); pr[te]=m.predict(sc.transform(X[te]))
        cccs.append(ccc(ys,pr)); maes.append(mean_absolute_error(ys,pr))
        try: aucs.append(roc_auc_score(yb,pr))
        except: pass
    print(f'  [{name:18s}] n={len(subs)} CCC={np.mean(cccs):.3f}±{np.std(cccs):.3f} MAE={np.mean(maes):.2f} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)

print('=== CMDC 앵커 조합 비교 (8 seed) ===',flush=True)
ev([3],'Q3 수면 단독')
ev([3,7],'Q3+Q7')
ev([3,7,1],'Q3+Q7+Q1 증상3')
ev([1,3,5,7],'증상4(1,3,5,7)')
ev(list(range(1,13)),'전체 12질문')
