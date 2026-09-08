"""
exp48 — LMVD(1823, 대형 in-the-wild, OpenFace AU) 재현.
같은 OpenFace AU로 정적 coupling + psychomotor. 라벨=ID범위.
결과: results/exp48_lmvd.csv
"""
import numpy as np, warnings, glob
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
V=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/LMVD/extracted/Video_feature')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
W=15;GAP=2
def lab(idn):
    if (1<=idn<=601) or (1117<=idn<=1423): return 1
    if (602<=idn<=1116) or (1425<=idn<=1824): return 0
    return None
def load(f):
    h=[x.strip() for x in open(f).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AU]
    except:return None
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if len(fe)>=40 else None
def psm(AUm):
    Z=(AUm-AUm.mean(0))/(AUm.std(0)+1e-6)
    spd=np.linalg.norm(np.diff(Z,axis=0),axis=1);g=spd.mean()+1e-6
    thr=np.percentile(spd,80);ev=[];t=W+GAP
    while t<len(spd)-1:
        if spd[t]>=thr and spd[t]>=spd[t-1] and spd[t]>=spd[t+1]:ev.append(t);t+=W
        else:t+=1
    pre=[spd[e-W-GAP:e-GAP].mean() for e in ev if e-W-GAP>=0];prem=np.mean(pre) if pre else g
    return np.array([spd.mean(),spd.std(),np.median(spd),len(ev)/len(spd),
                     np.mean([spd[e] for e in ev]) if ev else g, prem, prem/g,
                     np.mean([spd[e]-spd[e-W-GAP:e-GAP].mean() for e in ev if e-W-GAP>=0]) if pre else 0])
COV=[];PSM=[];y=[]
n=0
for f in sorted(glob.glob(str(V/'*.csv'))):
    idn=int(Path(f).stem); l=lab(idn)
    if l is None:continue
    au=load(f)
    if au is None:continue
    seg=(au-au.mean(0))/(au.std(0)+1e-6);c,_=ledoit_wolf(seg);COV.append(c+1e-4*np.eye(len(AU)))
    PSM.append(psm(au));y.append(l);n+=1
COV=np.array(COV);PSM=np.nan_to_num(np.array(PSM));y=np.array(y)
print(f'LMVD 로드 {n} (우울{y.sum()}/정상{(y==0).sum()})',flush=True)
print('=== psychomotor 기전(MDD vs HC) ===',flush=True)
for j,nm in enumerate(['spd_mean','spd_std','spd_med','rate','amp','pre_spd','pre_ratio','decel']):
    md,hc=PSM[y==1,j].mean(),PSM[y==0,j].mean();print(f'  {nm:9s} MDD={md:.3f} HC={hc:.3f} ({"↓" if md<hc else "↑"})',flush=True)
def cv(mode):
    a=[]
    for s in range(10):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(COV,y):
            if mode=='psm':F=PSM
            else:
                t=TangentSpace(metric='riemann').fit(COV[tr]);X=t.transform(COV)
                F=X if mode=='cov' else np.column_stack([X,PSM])
            sc=StandardScaler().fit(F[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(F[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(F[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
print('=== AUC (10seed CV) ===',flush=True)
rc=cv('cov');rp=cv('psm');rb=cv('comb')
print(f'  coupling   {rc[0]:.3f}±{rc[1]:.3f}',flush=True)
print(f'  psychomotor {rp[0]:.3f}±{rp[1]:.3f}',flush=True)
print(f'  결합       {rb[0]:.3f}±{rb[1]:.3f}',flush=True)
import csv as _c
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp48_lmvd.csv','w') as f:
    f.write('feature,cv_AUC,std\n')
    for nm,(m,sd) in [('coupling',rc),('psychomotor',rp),('combined',rb)]:f.write(f'{nm},{m:.4f},{sd:.4f}\n')
print('DONE → exp48_lmvd.csv',flush=True)
