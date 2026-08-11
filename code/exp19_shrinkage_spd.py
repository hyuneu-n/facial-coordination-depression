"""
exp19 — 제대로 된 SPD: Ledoit-Wolf shrinkage + 4자 대조 (자율탐색 조건).
CMDC 앵커(Q3+Q7). 대조: Euclidean(통계) / full-SPD(shrinkage) / random-window-SPD / anchor-SPD.
+ 공분산 조건수(왜 CMDC 되고 DAIC 안 되는지 정량) DAIC도 측정.
지표 AUC 10seed. shrinkage로 rank-deficient 방어.
"""
import numpy as np, csv, warnings, openpyxl
from pathlib import Path
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Shrinkage
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
SEEDS=list(range(10))
def lw_cov(X):
    """Ledoit-Wolf shrinkage 공분산 (rank-deficient 방어)"""
    if len(X)<3: return None
    c,_=ledoit_wolf(X); return c

# ---------- CMDC ----------
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU_C=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0]); iID,iMDD=hd.index('ID'),hd.index('MDD')
CL={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def c_q(subj,q):
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
    return np.array(fe) if len(fe)>=5 else None
def c_seq(subj,qs):
    parts=[c_q(subj,q) for q in qs]
    if any(p is None for p in parts): return None
    return np.vstack(parts)

def build_cmdc(mode):
    subs=[s for s in CL if c_seq(s,[3,7] if mode!='all' else list(range(1,13))) is not None]
    covs=[]; stats=[]; y=[]; conds=[]
    rng=np.random.RandomState(0)
    for s in subs:
        if mode=='random':
            allau=c_seq(s,list(range(1,13)))
            if allau is None or len(allau)<40: continue
            st=rng.randint(0,len(allau)-30); seg=allau[st:st+30]
        else:
            seg=c_seq(s,[3,7] if mode!='all' else list(range(1,13)))
        c=lw_cov(seg)
        if c is None: continue
        covs.append(c); stats.append(np.concatenate([seg.mean(0),seg.std(0)])); y.append(CL[s])
        conds.append(np.linalg.cond(c))
    return np.array(covs),np.array(stats),np.array(y),np.array(conds)

def auc_riemann(covs,y):
    aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y),float)
        for tr,te in skf.split(covs,y):
            clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(covs[tr],y[tr]); pb[te]=clf.decision_function(covs[te])
        aucs.append(roc_auc_score(y,pb))
    return np.array(aucs)
def auc_eucl(stats,y):
    aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y),float)
        for tr,te in skf.split(stats,y):
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(stats[tr],y[tr]); pb[te]=clf.decision_function(stats[te])
        aucs.append(roc_auc_score(y,pb))
    return np.array(aucs)

print('=== CMDC 4자 대조 (Ledoit-Wolf shrinkage SPD, 10seed) ===',flush=True)
cov_a,stat_a,y_a,cond_a=build_cmdc('anchor')
cov_all,stat_all,y_all,cond_all=build_cmdc('all')
cov_r,stat_r,y_r,cond_r=build_cmdc('random')
print(f'  1) Euclidean(통계) 앵커      AUC={auc_eucl(stat_a,y_a).mean():.3f}',flush=True)
print(f'  2) SPD 전체인터뷰            AUC={auc_riemann(cov_all,y_all).mean():.3f}',flush=True)
print(f'  3) SPD 랜덤윈도우            AUC={auc_riemann(cov_r,y_r).mean():.3f}',flush=True)
print(f'  4) SPD 앵커(제안)           AUC={auc_riemann(cov_a,y_a).mean():.3f}',flush=True)

# ---------- DAIC 공분산 조건수 (왜 안 되는지) ----------
D=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU_D=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
def d_lab():
    lab={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)): lab[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
    return lab
def d_tr(pid):
    p=D/f'{pid}_TRANSCRIPT.csv'; rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try: rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '')))
            except: pass
    return rows
def d_au(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists(): return None,None
    h=[x.strip() for x in open(p).readline().split(',')]
    ti,oi=h.index('timestamp'),h.index('success'); ai=[h.index(c) for c in AU_D]
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
POS=['last time you felt really happy','really happy','proud','enjoy','felt really happy']
dl=d_lab(); dcov=[]; dy=[]; dcond=[]
for pid,l in dl.items():
    ts,au=d_au(pid)
    if ts is None: continue
    segs=[]
    for st,sp,spk,val in d_tr(pid):
        if spk=='Ellie' and any(k in val.strip().lower() for k in POS):
            m=(ts>=sp)&(ts<sp+8.0)
            if m.sum()>=6: segs.append(au[m])
    if not segs: continue
    seg=np.vstack(segs); c=lw_cov(seg)
    if c is None: continue
    dcov.append(c); dy.append(l); dcond.append(np.linalg.cond(c))
dy=np.array(dy)
print(f'\n  DAIC 앵커 확보 n={len(dcov)}',flush=True)
if len(dcov)>=25:
    dcov=np.array(dcov)
    print(f'  DAIC SPD 앵커(shrinkage) AUC={auc_riemann(dcov,dy).mean():.3f}',flush=True)
else:
    print('  DAIC 앵커 세그먼트 부족 — 재현 확인 필요',flush=True)
print(f'\n=== 공분산 조건수 (왜 CMDC>DAIC) ===',flush=True)
print(f'  CMDC 앵커 평균 세그길이/조건수: cond중앙값={np.median(cond_a):.1f}',flush=True)
print(f'  DAIC 앵커 조건수 중앙값={np.median(dcond):.1f}',flush=True)
print(f'  → 조건수 클수록 rank-deficient/불안정. (shrinkage 후 비교)',flush=True)
