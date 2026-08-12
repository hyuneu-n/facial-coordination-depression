"""
exp31 — whole vs anchor, 최종 파이프라인(공분산 Ledoit-Wolf → Riemannian tangent → 로지스틱)으로 동일하게.
G2(앵커효과)를 G1(메인결과)과 apples-to-apples로 만들기 위함. 같은 subject set 사용.
CMDC: whole(Q1~12) vs anchor(Q3+7). DAIC: whole(전체 success) vs anchor(ALL_neg).
결과 CSV: results/whole_vs_anchor.csv
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
def evalcov(C,y,name):
    C=np.array(C);y=np.array(y);aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            t=TangentSpace(metric='riemann').fit(C[tr]);Xtr=t.transform(C[tr]);Xte=t.transform(C[te])
            sc=StandardScaler().fit(Xtr)
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),y[tr])
            pb[te]=clf.decision_function(sc.transform(Xte))
        aucs.append(roc_auc_score(y,pb))
    print(f'  {name:22s} n={len(y)} 우울{int(sum(y))} AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}',flush=True)
    return name,len(y),int(sum(y)),float(np.mean(aucs)),float(np.std(aucs))

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
Cw=[];Ca=[];yw=[];ya=[]
for s,l in cl.items():
    anch=[cq(s,q) for q in [3,7]]
    if not all(x is not None for x in anch): continue   # 앵커 가능한 subject만 = 동일 set
    whole=[cq(s,q) for q in range(1,13)]; whole=[x for x in whole if x is not None]
    if not whole: continue
    sa=np.vstack(anch); sa=(sa-sa.mean(0))/(sa.std(0)+1e-6); ca,_=ledoit_wolf(sa); Ca.append(ca); ya.append(l)
    sw=np.vstack(whole); sw=(sw-sw.mean(0))/(sw.std(0)+1e-6); cw,_=ledoit_wolf(sw); Cw.append(cw); yw.append(l)
print('=== CMDC whole vs anchor (동일 파이프라인) ===',flush=True)
results.append(('CMDC_whole',)+evalcov(Cw,yw,'CMDC whole(Q1-12)')[1:])
results.append(('CMDC_anchor',)+evalcov(Ca,ya,'CMDC anchor(Q3+7)')[1:])

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
Cdw=[];Cda=[];ydw=[];yda=[]
for pid,l in dl.items():
    ts,au=dseries(pid)
    if ts is None:continue
    segs=[]
    for st,sp,spk,val in dtrans(pid):
        if spk=='Ellie' and any(val.startswith(t) for t in NEG):
            m=(ts>=sp)&(ts<sp+8.0)
            if m.sum()>=8:segs.append(au[m])
    if not segs:continue
    sa=np.vstack(segs);sa=(sa-sa.mean(0))/(sa.std(0)+1e-6)
    if len(sa)<15:continue
    ca,_=ledoit_wolf(sa);Cda.append(ca);yda.append(l)
    sw=(au-au.mean(0))/(au.std(0)+1e-6); cw,_=ledoit_wolf(sw); Cdw.append(cw); ydw.append(l)  # whole = 전체 success
print('=== DAIC whole vs anchor (동일 파이프라인) ===',flush=True)
results.append(('DAIC_whole',)+evalcov(Cdw,ydw,'DAIC whole(all)')[1:])
results.append(('DAIC_anchor',)+evalcov(Cda,yda,'DAIC anchor(ALLneg)')[1:])

with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/whole_vs_anchor.csv','w') as f:
    f.write('key,n,depressed,AUC,std\n')
    for k,n,d,a,s in results:f.write(f'{k},{n},{d},{a:.4f},{s:.4f}\n')
print('\nDONE → whole_vs_anchor.csv',flush=True)
