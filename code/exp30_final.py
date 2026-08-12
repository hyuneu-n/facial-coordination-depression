"""
exp30 — 확정 방법을 전 데이터 통일 세팅으로 최종 실행. 발표용 최종 표.
방법: 앵커 구간 AU 공분산(Ledoit-Wolf) → Riemannian tangent → 로지스틱.
CMDC(앵커 Q3+Q7), DAIC(앵커 ALL_neg 질문), E-DAIC(품질게이트 frontal+conf, 앵커불가).
각: AUC 10seed + permutation p(100). 결과 CSV.
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
SEEDS=list(range(10))
def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean();vy,vp=y.var(),yp.var();cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)
def evalcov(C,y,name,perm=True):
    C=np.array(C);y=np.array(y);aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            t=TangentSpace(metric='riemann').fit(C[tr]);Xtr=t.transform(C[tr]);Xte=t.transform(C[te])
            sc=StandardScaler().fit(Xtr)
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),y[tr])
            pb[te]=clf.decision_function(sc.transform(Xte))
        aucs.append(roc_auc_score(y,pb))
    obs=np.mean(aucs);p=np.nan
    if perm:
        rng=np.random.RandomState(0);null=[]
        for _ in range(100):
            yp=rng.permutation(y);skf=StratifiedKFold(5,shuffle=True,random_state=42);pb=np.zeros(len(y))
            for tr,te in skf.split(C,yp):
                t=TangentSpace(metric='riemann').fit(C[tr]);Xtr=t.transform(C[tr]);Xte=t.transform(C[te])
                sc=StandardScaler().fit(Xtr)
                clf=LogisticRegression(max_iter=1000,class_weight='balanced').fit(sc.transform(Xtr),yp[tr])
                pb[te]=clf.decision_function(sc.transform(Xte))
            null.append(roc_auc_score(yp,pb))
        p=(np.sum(np.array(null)>=obs)+1)/101
    print(f'  {name:18s} n={len(y)} 우울{int(sum(y))} AUC={obs:.3f}±{np.std(aucs):.3f} perm_p={p:.3f}',flush=True)
    return name,len(y),int(sum(y)),obs,np.std(aucs),p

results=[]
# ===== CMDC =====
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
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
print('=== 최종 통합 (앵커+공분산 Riemannian) ===',flush=True)
results.append(evalcov(Cc,yc,'CMDC(앵커Q3+7)'))

# ===== DAIC =====
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
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
results.append(evalcov(Cd,yd,'DAIC(앵커ALLneg)'))

# ===== E-DAIC (품질게이트, 앵커불가) =====
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
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
results.append(evalcov(Ce,ye,'E-DAIC(품질게이트)',perm=True))

with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/FINAL_table.csv','w') as f:
    f.write('dataset,n,depressed,AUC,std,perm_p\n')
    for nm,n,d,a,s,p in results:f.write(f'{nm},{n},{d},{a:.4f},{s:.4f},{p:.4f}\n')
print('\nDONE → FINAL_table.csv',flush=True)
