"""
exp29 — DAIC-WOZ에 CMDC와 완전 동일 pipeline (공분산 Riemannian + 앵커).
exp14(대충 0.51)와 달리 질문 앵커 조합 최적화 + frontal 필터 없이 CLNF success.
여러 질문 앵커별 + 조합 공분산 → tangent → 로지스틱. 손절 0.7.
DAIC는 CLNF(14 AU), 질문 태그 transcript 있음.
"""
import numpy as np, warnings, csv
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
D=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
POST=8.0; SEEDS=list(range(10))
# 질문 앵커 그룹 (증상/부정정서 위주 = CMDC 교훈)
GROUPS={
 'sleep':['easy_sleep','sleep_affects'],
 'mood':['feel_lately'],
 'positive':['happy_lasttime','dream_job'],
 'depression':['depression_diagnosed'],
 'guilt_regret':['feelguilty','regret','feelbadly'],
 'ALL_symptom':['easy_sleep','sleep_affects','feel_lately','depression_diagnosed','feelguilty','regret'],
 'ALL_neg':['feel_lately','depression_diagnosed','feelguilty','regret','feelbadly','last_argument','control_temper'],
}
def labels():
    lab={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)): lab[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
    return lab
def transcript(pid):
    p=D/f'{pid}_TRANSCRIPT.csv'; rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try: rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except: pass
    return rows
def au_series(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists(): return None,None
    h=[x.strip() for x in open(p).readline().split(',')]
    ti,oi=h.index('timestamp'),h.index('success'); ai=[h.index(c) for c in AU]
    ts,fe=[],[]
    with open(p) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                ts.append(float(v[ti])); fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(ts),np.array(fe)
lab=labels()
DATA={}
for pid in lab:
    ts,au=au_series(pid)
    if ts is not None: DATA[pid]=(ts,au,transcript(pid))
def cov_of(pid,tags):
    ts,au,rows=DATA[pid]
    segs=[]
    for st,sp,spk,val in rows:
        if spk=='Ellie' and any(val.startswith(t) for t in tags):
            m=(ts>=sp)&(ts<sp+POST)
            if m.sum()>=8: segs.append(au[m])
    if not segs: return None
    seg=np.vstack(segs); seg=(seg-seg.mean(0))/(seg.std(0)+1e-6)
    if len(seg)<15: return None
    c,_=ledoit_wolf(seg); return c
def evalg(tags,name):
    C=[];y=[]
    for pid in DATA:
        c=cov_of(pid,tags)
        if c is not None: C.append(c);y.append(lab[pid])
    if len(y)<30 or sum(y)<8: print(f'  {name:16s} 표본부족 n={len(y)}',flush=True); return name,len(y),0
    C=np.array(C);y=np.array(y); aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            ts_=TangentSpace(metric='riemann').fit(C[tr]);Xtr=ts_.transform(C[tr]);Xte=ts_.transform(C[te])
            sc=StandardScaler().fit(Xtr)
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),y[tr])
            pb[te]=clf.decision_function(sc.transform(Xte))
        aucs.append(roc_auc_score(y,pb))
    print(f'  {name:16s} n={len(y)} 우울{int(y.sum())} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)
    return name,len(y),np.mean(aucs)
print('=== DAIC 공분산 Riemannian + 앵커 (CMDC와 동일 pipeline) ===',flush=True)
res=[]
for g,tags in GROUPS.items():
    res.append(evalg(tags,g))
best=max(res,key=lambda r:r[2])
print(f'\n[최고] {best[0]} AUC={best[2]:.3f} → {"성공(>0.7)✅" if best[2]>=0.7 else "미달"}',flush=True)
print('(CMDC 공분산 0.885 / DAIC 기존 ST-GCN 0.61)',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/daic_cov.csv','w') as f:
    f.write('anchor,n,AUC\n')
    for nm,n,a in res: f.write(f'{nm},{n},{a:.4f}\n')
print('DONE',flush=True)
