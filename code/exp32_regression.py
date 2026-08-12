"""
exp32 — severity regression (CCC + MAE), 최종 coupling 파이프라인으로.
분류(exp30)와 동일한 특징: anchor 구간 AU covariance(Ledoit-Wolf) → Riemannian tangent.
분류기 대신 RidgeCV 로 PHQ 점수(연속) 예측. AVEC 공식지표 CCC + MAE + RMSE.
CMDC(PHQtotal), DAIC-WOZ(PHQ8_Score), E-DAIC(PHQ_Score). 10-seed KFold + permutation p(CCC).
결과 CSV: results/FINAL_regression.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
SEEDS=list(range(10)); ALPHAS=[1.0,10.0,100.0]
def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean();vy,vp=y.var(),yp.var();cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)
def evalreg(C,y,name,perm=True):
    C=np.array(C);y=np.array(y,float);cccs=[];maes=[];rmses=[]
    for sd in SEEDS:
        kf=KFold(5,shuffle=True,random_state=sd);pr=np.zeros(len(y))
        for tr,te in kf.split(C):
            t=TangentSpace(metric='riemann').fit(C[tr]);Xtr=t.transform(C[tr]);Xte=t.transform(C[te])
            sc=StandardScaler().fit(Xtr)
            reg=RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr),y[tr])
            pr[te]=reg.predict(sc.transform(Xte))
        cccs.append(ccc(y,pr));maes.append(mean_absolute_error(y,pr));rmses.append(np.sqrt(mean_squared_error(y,pr)))
    obs=np.mean(cccs);p=np.nan
    if perm:
        rng=np.random.RandomState(0);null=[]
        for _ in range(100):
            yp=rng.permutation(y);kf=KFold(5,shuffle=True,random_state=42);pr=np.zeros(len(y))
            for tr,te in kf.split(C):
                t=TangentSpace(metric='riemann').fit(C[tr]);Xtr=t.transform(C[tr]);Xte=t.transform(C[te])
                sc=StandardScaler().fit(Xtr)
                reg=RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr),yp[tr]);pr[te]=reg.predict(sc.transform(Xte))
            null.append(ccc(yp,pr))
        p=(np.sum(np.array(null)>=obs)+1)/101
    print(f'  {name:20s} n={len(y)} CCC={obs:.3f}±{np.std(cccs):.3f} MAE={np.mean(maes):.2f} RMSE={np.mean(rmses):.2f} perm_p={p:.3f}',flush=True)
    return name,len(y),float(obs),float(np.std(cccs)),float(np.mean(maes)),float(np.mean(rmses)),float(p)

results=[]
# ===== CMDC (PHQtotal) =====
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID=hd.index('ID');iP=[hd.index(f'PHQ-{i}') for i in range(1,10)]
cl={}
for r in rows[1:]:
    if r[iID] is None:continue
    try:cl[str(r[iID]).strip()]=float(sum(int(r[i]) for i in iP if r[i] is not None))
    except:pass
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if len(fe)>=10 else None
Cc=[];yc=[]
for s,l in cl.items():
    segs=[cq(s,q) for q in [3,7]]
    if all(x is not None for x in segs):
        seg=np.vstack(segs);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);Cc.append(c);yc.append(l)
print('=== severity regression (CCC/MAE), 최종 coupling 파이프라인 ===',flush=True)
results.append(evalreg(Cc,yc,'CMDC(anchor Q3+7)'))

# ===== DAIC-WOZ (PHQ8_Score) =====
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=float(r['PHQ8_Score'])
NEG=['feel_lately','depression_diagnosed','feelguilty','regret','feelbadly','last_argument','control_temper']
def dseries(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None,None
    h=[x.strip() for x in open(p).readline().split(',')];ti,oi=h.index('timestamp'),h.index('success');ai=[h.index(c) for c in AUd]
    ts,fe=[],[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            ts.append(float(v[ti]));fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(ts),np.array(fe)
def dtrans(pid):
    p=D/f'{pid}_TRANSCRIPT.csv';rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try:rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except:pass
    return rows
Cd=[];yd=[]
for pid,l in dl.items():
    ts,au=dseries(pid)
    if ts is None:continue
    segs=[]
    for st,sp,spk,val in dtrans(pid):
        if spk=='Ellie' and any(val.startswith(t) for t in NEG):
            m=(ts>=sp)&(ts<sp+8.0)
            if m.sum()>=8:segs.append(au[m])
    if not segs:continue
    seg=np.vstack(segs);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6)
    if len(seg)<15:continue
    c,_=ledoit_wolf(seg);Cd.append(c);yd.append(l)
results.append(evalreg(Cd,yd,'DAIC-WOZ(anchor)'))

# ===== E-DAIC (PHQ_Score, quality gate) =====
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();s=r.get('PHQ_Score') or r.get('PHQ8_Score')
            if s not in(None,''):el[pid]=float(s)
Ce=[];ye=[]
for pid,l in el.items():
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
    if not fs:continue
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc];rx,ry,rz=h.index('pose_Rx'),h.index('pose_Ry'),h.index('pose_Rz')
    except:continue
    seg=[]
    for ln in open(fs[0]).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            if max(abs(float(v[rx])),abs(float(v[ry])),abs(float(v[rz])))>0.35:continue
            seg.append([float(v[i]) for i in ai])
        except:pass
    if len(seg)<20:continue
    seg=np.array(seg);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);Ce.append(c);ye.append(l)
results.append(evalreg(Ce,ye,'E-DAIC(quality gate)'))

with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/FINAL_regression.csv','w') as f:
    f.write('dataset,n,CCC,CCC_std,MAE,RMSE,perm_p\n')
    for nm,n,c,cs,m,rm,p in results:f.write(f'{nm},{n},{c:.4f},{cs:.4f},{m:.4f},{rm:.4f},{p:.4f}\n')
print('\nDONE → FINAL_regression.csv',flush=True)
