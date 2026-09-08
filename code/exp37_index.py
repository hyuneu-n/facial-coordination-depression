"""
exp37 — Facial Coupling Disruption Index (FCDI): 우리 방법을 1-D 스칼라 지수로.
각 사람 anchored 공분산 Σ_i → 훈련폴드 HC 평균 공분산에서 Riemannian 거리 = 지수.
누수 방지 CV(HC평균은 train만). 지수 단독 AUC + PHQ 상관(Spearman). CMDC, DAIC.
결과: results/coupling_index.csv
"""
import numpy as np, warnings, csv, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=list(range(10))
def eval_index(C,y,phq,name):
    C=np.array(C);y=np.array(y);phq=np.array(phq,float);aucs=[]
    dist_all=np.zeros(len(y))  # 1 seed로 상관용 거리 축적
    for si,sd in enumerate(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=sd);d=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            hc=C[tr][y[tr]==0]
            ref=mean_riemann(hc)
            for i in te: d[i]=distance_riemann(C[i],ref)
        aucs.append(roc_auc_score(y,d))
        if si==0: dist_all=d.copy()
    rho,p=spearmanr(dist_all,phq)
    print(f'  {name:16s} n={len(y)} FCDI-AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  Spearman(지수,PHQ)={rho:.3f}(p={p:.3g})',flush=True)
    return name,len(y),float(np.mean(aucs)),float(np.std(aucs)),float(rho),float(p)

res=[]
# CMDC
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD');iP=[hd.index(f'PHQ-{i}') for i in range(1,10)]
lab={}
for r in rows[1:]:
    if r[iID] is None:continue
    try:tot=sum(int(r[i]) for i in iP if r[i] is not None)
    except:continue
    lab[str(r[iID]).strip()]=(int(r[iMDD]),tot)
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
Cc=[];yc=[];pc=[]
for s,(l,tot) in lab.items():
    segs=[cq(s,q) for q in [3,7]]
    if all(x is not None for x in segs):
        seg=np.vstack(segs);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);Cc.append(c);yc.append(l);pc.append(tot)
print('=== Facial Coupling Disruption Index (HC평균에서 Riemannian 거리) ===',flush=True)
res.append(eval_index(Cc,yc,pc,'CMDC'))

# DAIC
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=(int(float(r['PHQ8_Binary'])),float(r['PHQ8_Score']))
NEG=['feel_lately','depression_diagnosed','feelguilty','regret','feelbadly','last_argument','control_temper']
def dser(pid):
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
def dtr(pid):
    p=D/f'{pid}_TRANSCRIPT.csv';rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try:rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except:pass
    return rows
Cd=[];yd=[];pd=[]
for pid,(l,tot) in dl.items():
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
    c,_=ledoit_wolf(seg);Cd.append(c);yd.append(l);pd.append(tot)
res.append(eval_index(Cd,yd,pd,'DAIC-WOZ'))

with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/coupling_index.csv','w') as f:
    f.write('dataset,n,FCDI_AUC,std,spearman_PHQ,p\n')
    for nm,n,a,s,rho,p in res: f.write(f'{nm},{n},{a:.4f},{s:.4f},{rho:.4f},{p:.4g}\n')
print('\nDONE (참고: 분류기 full-method AUC = CMDC 0.889 / DAIC 0.711)',flush=True)
