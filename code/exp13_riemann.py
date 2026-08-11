"""
exp13 — AU 공분산 → Riemannian (EEG 기법 이식). CMDC 앵커 + DAIC.
방법: 앵커 구간 AU 시계열 → 공분산 SPD → (a)접공간+로지스틱 분류 (b)접공간+Ridge 회귀(CCC).
baseline: CMDC Ridge 앵커 CCC0.51/AUC0.81. 비교.
"""
import numpy as np, csv, warnings, openpyxl
from pathlib import Path
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error
warnings.filterwarnings('ignore')
SEEDS=[42,1,2,3,4]
def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean();vy,vp=y.var(),yp.var();cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)

# ---------- CMDC ----------
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU_C=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0])
iID,iMDD=hd.index('ID'),hd.index('MDD'); iP=[hd.index(f'PHQ-{i}') for i in range(1,10)]
clab={}
for r in rows[1:]:
    if r[iID] is None: continue
    try: tot=sum(int(r[i]) for i in iP if r[i] is not None)
    except: continue
    clab[str(r[iID]).strip()]=(tot,int(r[iMDD]))
def cmdc_q(subj,q):
    f=C/subj/f'Q{q}.csv'
    if not f.exists(): return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try: oi=h.index('success'); ai=[h.index(c) for c in AU_C]
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
def cmdc_seq(subj,qs):
    parts=[cmdc_q(subj,q) for q in qs]
    if any(p is None for p in parts): return None
    return np.vstack(parts).T  # (channels, time) for pyriemann

def run_riemann(Xlist,ys,yb,name):
    # Xlist: list of (C,T) arrays
    ncov=[]
    # 공분산 추정 (pyriemann은 (n_trials, n_channels, n_times) 요구 → 길이 맞춰 자름/패딩 대신 개별 공분산)
    covs=[]
    for x in Xlist:
        c=np.cov(x)  # (C,C)
        c=c+1e-6*np.eye(c.shape[0])  # SPD 보정
        covs.append(c)
    covs=np.array(covs)
    ys=np.array(ys); yb=np.array(yb)
    # 분류 AUC
    aucs=[]; cccs=[]; maes=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(yb),float)
        for tr,te in skf.split(covs,yb):
            clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(covs[tr],yb[tr]); pb[te]=clf.decision_function(covs[te])
        try: aucs.append(roc_auc_score(yb,pb))
        except: pass
        kf=KFold(5,shuffle=True,random_state=sd); pr=np.zeros(len(ys),float)
        for tr,te in kf.split(covs):
            reg=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),Ridge(alpha=10))
            reg.fit(covs[tr],ys[tr]); pr[te]=reg.predict(covs[te])
        cccs.append(ccc(ys,pr)); maes.append(mean_absolute_error(ys,pr))
    print(f'  [{name:20s}] n={len(ys)} CCC={np.mean(cccs):.3f}±{np.std(cccs):.3f} MAE={np.mean(maes):.2f} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)

print('=== Riemannian (AU 공분산) — CMDC ===',flush=True)
for qs,nm in [([3,7],'앵커 Q3+Q7'),(list(range(1,13)),'전체12')]:
    subs=[s for s in clab if cmdc_seq(s,qs) is not None]
    X=[cmdc_seq(s,qs) for s in subs]; ys=[clab[s][0] for s in subs]; yb=[clab[s][1] for s in subs]
    run_riemann(X,ys,yb,nm)
print('  (baseline: Ridge 앵커 CCC0.51/AUC0.81)',flush=True)
