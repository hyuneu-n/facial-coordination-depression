"""
exp9 — CMDC(중국, 45명 표정) 스모크. 지표 CCC/MAE/AUC.
질문별 파일(Q1~Q12) = 앵커 자연스러움. 먼저 전체(모든 Q) → 그다음 Q별.
PHQtotal 회귀 + 이진(MDD).
"""
import numpy as np, glob, warnings, openpyxl
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
warnings.filterwarnings('ignore')
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
# CMDC AU 열 (35개 중 _r 강도만)
AU_R=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r',
      'AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
SEEDS=[42,1,2,3,4]

def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean();vy,vp=y.var(),yp.var();cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)

# 라벨
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0])
iID,iMDD=hd.index('ID'),hd.index('MDD'); iP=[hd.index(f'PHQ-{i}') for i in range(1,10)]
lab={}
for r in rows[1:]:
    if r[iID] is None: continue
    try: tot=sum(int(r[i]) for i in iP if r[i] is not None)
    except: continue
    lab[str(r[iID]).strip()]=(tot,int(r[iMDD]))

def au_of(subj):
    """참가자의 모든 Q AU 이어붙임 (전체)"""
    feats=[]
    for q in range(1,13):
        f=C/subj/f'Q{q}.csv'
        if not f.exists(): continue
        h=[x.strip() for x in open(f).readline().split(',')]
        try: oi=h.index('success'); ai=[h.index(c) for c in AU_R]
        except: continue
        with open(f) as fp:
            fp.readline()
            for ln in fp:
                v=ln.split(',')
                try:
                    if int(float(v[oi]))!=1: continue
                    feats.append([float(v[i]) for i in ai])
                except: pass
    return np.array(feats) if len(feats)>10 else None

data={}
for subj in lab:
    au=au_of(subj)
    if au is not None: data[subj]=au
print(f'CMDC 로드: {len(data)}명 (표정 있는), PHQ평균 {np.mean([lab[s][0] for s in data]):.1f}, MDD {sum(lab[s][1] for s in data)}\n',flush=True)
if len(data)<20: print('데이터 부족'); raise SystemExit

def statfeat(au): return np.concatenate([au.mean(0),au.std(0)])
def lowdim(au):
    X=au-au.mean(0,keepdims=True); s=np.linalg.svd(X,compute_uv=False); s=s[s>1e-8]
    if len(s)<2: return np.zeros(3)
    p=s/s.sum();s2=s**2
    return np.array([p[0],(s2.sum()**2)/((s2**2).sum()),np.exp(-(p*np.log(p+1e-12)).sum())])

def ev(featfn,name):
    subs=list(data); X=np.array([featfn(data[s]) for s in subs])
    ys=np.array([lab[s][0] for s in subs]); yb=np.array([lab[s][1] for s in subs])
    cccs,maes,aucs=[],[],[]
    for sd in SEEDS:
        kf=KFold(5,shuffle=True,random_state=sd); pr=np.zeros(len(ys))
        for tr,te in kf.split(X):
            sc=StandardScaler().fit(X[tr]); m=Ridge(alpha=10).fit(sc.transform(X[tr]),ys[tr])
            pr[te]=m.predict(sc.transform(X[te]))
        cccs.append(ccc(ys,pr)); maes.append(mean_absolute_error(ys,pr))
        try: aucs.append(roc_auc_score(yb,pr))
        except: pass
    print(f'  [{name}] CCC={np.mean(cccs):.3f} MAE={np.mean(maes):.2f} AUC={np.mean(aucs):.3f}',flush=True)

print('=== CMDC 전체(모든 Q, 앵커無) CCC/MAE/AUC ===',flush=True)
ev(statfeat,'AU통계'); ev(lowdim,'저차원성'); ev(lambda a:np.concatenate([statfeat(a),lowdim(a)]),'통계+저차원')
