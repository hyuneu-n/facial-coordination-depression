"""
exp41 (Plan A 핵심) — 교차 코퍼스 coupling transfer: 학습(코퍼스A)→테스트(코퍼스B).
공통 14 AU로 CMDC(중)·DAIC(영) anchor covariance. within-corpus CV + cross-corpus transfer AUC.
transfer: TangentSpace(ref=train corpus mean)로 양쪽 사상 → train logistic → test 적용.
= "언어 넘어 되는 해석가능 coupling 마커"의 실증.
결과: results/exp41_crosscorpus.csv
"""
import numpy as np, warnings, csv, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=list(range(10))
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']  # 공통 14

# ---- CMDC ----
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AU]
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
Cc=np.array(Cc);yc=np.array(yc)

# ---- DAIC ----
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
NEG=['feel_lately','depression_diagnosed','feelguilty','regret','feelbadly','last_argument','control_temper']
def dser(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None,None
    h=[x.strip() for x in open(p).readline().split(',')];ti,oi=h.index('timestamp'),h.index('success');ai=[h.index(c) for c in AU]
    ts,fe=[],[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            ts.append(float(v[ti]));fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(ts),np.array(fe)
def dtr(pid):
    p=D/f'{pid}_TRANSCRIPT.csv';r_=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try:r_.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except:pass
    return r_
Cd=[];yd=[]
for pid,l in dl.items():
    ts,au=dser(pid)
    if ts is None:continue
    segs=[]
    for st,sp,spk,val in dtr(pid):
        if spk=='Ellie' and any(val.startswith(t) for t in NEG):
            m=(ts>=sp)&(ts<sp+8.0)
            if m.sum()>=8:segs.append(au[m])
    if not segs:continue
    seg=np.vstack(segs);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6)
    if len(seg)<15:continue
    c,_=ledoit_wolf(seg);Cd.append(c);yd.append(l)
Cd=np.array(Cd);yd=np.array(yd)
print(f'CMDC n={len(yc)}(dep{yc.sum()}) / DAIC n={len(yd)}(dep{yd.sum()}) · 공통 {len(AU)} AU',flush=True)

def within(C,y,name):
    a=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            t=TangentSpace(metric='riemann').fit(C[tr]);Xtr=t.transform(C[tr]);Xte=t.transform(C[te])
            s=StandardScaler().fit(Xtr);clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(s.transform(Xtr),y[tr])
            pb[te]=clf.decision_function(s.transform(Xte))
        a.append(roc_auc_score(y,pb))
    print(f'  within {name}: AUC={np.mean(a):.3f}±{np.std(a):.3f}',flush=True); return np.mean(a),np.std(a)
def transfer(Ctr,ytr,Cte,yte,name):
    t=TangentSpace(metric='riemann').fit(Ctr)              # ref=train corpus
    Xtr=t.transform(Ctr);Xte=t.transform(Cte)
    s=StandardScaler().fit(Xtr)
    clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(s.transform(Xtr),ytr)
    auc=roc_auc_score(yte,clf.decision_function(s.transform(Xte)))
    print(f'  transfer {name}: AUC={auc:.3f}',flush=True); return auc

print('=== within-corpus (공통14 AU) ===',flush=True)
wc=within(Cc,yc,'CMDC'); wd=within(Cd,yd,'DAIC')
print('=== cross-corpus transfer (일반화 증거) ===',flush=True)
t_cd=transfer(Cc,yc,Cd,yd,'CMDC→DAIC')
t_dc=transfer(Cd,yd,Cc,yc,'DAIC→CMDC')
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp41_crosscorpus.csv','w') as f:
    f.write('metric,AUC\n')
    f.write(f'within_CMDC,{wc[0]:.4f}\nwithin_DAIC,{wd[0]:.4f}\ntransfer_CMDC2DAIC,{t_cd:.4f}\ntransfer_DAIC2CMDC,{t_dc:.4f}\n')
print('\nDONE → exp41_crosscorpus.csv',flush=True)
