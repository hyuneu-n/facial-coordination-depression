"""
exp17 — 교차언어 ADI 순위 상관 (실험3, 불변성 정식 판정).
DAIC/CMDC 질문을 PHQ 증상축에 매핑 → 증상축별 진단력(AUC) → 두 데이터 순위 Spearman ρ + permutation.
공통 증상축이 적으면 검정력 약함 = 정직하게 명시.
"""
import numpy as np, csv, warnings, openpyxl
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
SEEDS=list(range(10))

# ===== 증상축 매핑 =====
# DAIC 질문태그 → PHQ 증상축
DAIC_AXIS={'easy_sleep':'수면','sleep_affects':'수면','feel_lately':'기분','dream_job':'흥미',
           'happy_lasttime':'흥미','feelguilty':'자기비하','memory_erase':'자기비하','regret':'자기비하'}
# CMDC Q번호 → PHQ 증상축
CMDC_AXIS={1:'식욕',3:'수면',5:'집중',6:'집중',7:'피로',8:'자해',9:'기분',12:'기분'}

# ---------- DAIC ----------
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
DL=d_lab(); DDATA={}
for pid in DL:
    ts,au=d_au(pid)
    if ts is not None: DDATA[pid]=(ts,au,d_tr(pid))
def d_feat_axis(pid,axis):
    ts,au,rows=DDATA[pid]; base=au.mean(0); bs=au.std(0)+1e-6
    tags=[t for t,ax in DAIC_AXIS.items() if ax==axis]
    segs=[]
    for st,sp,spk,val in rows:
        if spk=='Ellie' and any(val.strip().lower().startswith(t) for t in tags):
            m=(ts>=sp)&(ts<sp+8.0)
            if m.sum()>=6: segs.append(au[m])
    if not segs: return None
    return np.concatenate([( (np.vstack(segs)-base)/bs ).mean(0),((np.vstack(segs)-base)/bs).std(0)])

# ---------- CMDC ----------
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU_C=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0])
iID,iMDD=hd.index('ID'),hd.index('MDD')
CL={}
for r in rows[1:]:
    if r[iID] is None: continue
    CL[str(r[iID]).strip()]=int(r[iMDD])
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
def c_feat_axis(subj,axis):
    qs=[q for q,ax in CMDC_AXIS.items() if ax==axis]
    segs=[c_q(subj,q) for q in qs]; segs=[s for s in segs if s is not None]
    if not segs: return None
    cat=np.vstack(segs)
    return np.concatenate([cat.mean(0),cat.std(0)])

def auc_axis(feat_fn,lab,axis):
    F,y=[],[]
    for pid,l in lab.items():
        if feat_fn is d_feat_axis and pid not in DDATA: continue
        f=feat_fn(pid,axis)
        if f is not None: F.append(f); y.append(l)
    if len(y)<25 or sum(y)<6: return None
    F,y=np.array(F),np.array(y); aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y),float)
        for tr,te in skf.split(F,y):
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(F[tr],y[tr]); pb[te]=clf.decision_function(F[te])
        aucs.append(roc_auc_score(y,pb))
    return np.mean(aucs)

axes=set(DAIC_AXIS.values())|set(CMDC_AXIS.values())
print('=== 증상축별 진단력 (DAIC vs CMDC) ===',flush=True)
print(f"{'증상축':>8} {'DAIC':>7} {'CMDC':>7}",flush=True)
d_scores={}; c_scores={}
for ax in sorted(axes):
    da=auc_axis(d_feat_axis,DL,ax); ca=auc_axis(c_feat_axis,CL,ax)
    ds=f"{da:.3f}" if da else "  -  "; cs=f"{ca:.3f}" if ca else "  -  "
    print(f"{ax:>8} {ds:>7} {cs:>7}",flush=True)
    if da is not None: d_scores[ax]=da
    if ca is not None: c_scores[ax]=ca
common=sorted(set(d_scores)&set(c_scores))
print(f"\n공통 증상축: {common} ({len(common)}개)",flush=True)
if len(common)>=3:
    dv=[d_scores[a] for a in common]; cv=[c_scores[a] for a in common]
    rho,p=spearmanr(dv,cv)
    print(f"Spearman ρ={rho:.3f}, p={p:.3f} (n={len(common)})",flush=True)
    print(f"→ 불변성 {'지지' if rho>0.5 and p<0.1 else '약함/불명확'}",flush=True)
else:
    print(f"공통 증상축 {len(common)}개 → 상관검정 불가(검정력 없음). 불변성 정식검증 불가.",flush=True)
