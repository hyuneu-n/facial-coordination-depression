"""
exp15 — 앵커 효과 정식 증명. 앵커 有 vs 無 × 여러 방법 × 2 데이터 + 통계.
주장: "앵커(질문 선택)가 표정 기반 우울 판별을 일관되게 향상시킨다."
방법: 경량(Ridge회귀/LogReg분류) — 데이터무관 안정. 특징=AU 평균+std.
데이터: DAIC(긍정앵커 vs 전체), CMDC(증상앵커 Q3+Q7 vs 전체).
통계: 앵커有 vs 無 AUC 차이 순열검정(subject 라벨고정, fold seed 여러개 → 쌍체 비교).
지표: AUC(분류) + CCC(회귀).
"""
import numpy as np, csv, warnings, openpyxl
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')
SEEDS=list(range(10))  # 10 seed로 쌍체 통계
def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean();vy,vp=y.var(),yp.var();cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)
def statfeat(au): return np.concatenate([au.mean(0),au.std(0)])

# ---------- DAIC ----------
D=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU_D=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
POS=['last time you felt really happy','really happy','proud','enjoy','felt really happy']
def d_lab():
    lab={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)): lab[r['Participant_ID'].strip()]=(float(r['PHQ8_Score']),int(float(r['PHQ8_Binary'])))
    return lab
def d_tr(pid):
    p=D/f'{pid}_TRANSCRIPT.csv'; rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try: rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
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
def d_feat(pid,mode):
    ts,au=d_au(pid)
    if ts is None: return None
    if mode=='all': seg=au
    else:
        anch=[sp for st,sp,spk,val in d_tr(pid) if spk=='Ellie' and any(k in val for k in POS)]
        segs=[au[(ts>=a)&(ts<a+6.0)] for a in anch]; segs=[s for s in segs if len(s)>=6]
        if not segs: return None
        seg=np.vstack(segs)
    base=au.mean(0); bs=au.std(0)+1e-6
    return statfeat((seg-base)/bs)

# ---------- CMDC ----------
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU_C=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0])
iID,iMDD=hd.index('ID'),hd.index('MDD'); iP=[hd.index(f'PHQ-{i}') for i in range(1,10)]
c_lab={}
for r in rows[1:]:
    if r[iID] is None: continue
    try: tot=sum(int(r[i]) for i in iP if r[i] is not None)
    except: continue
    c_lab[str(r[iID]).strip()]=(tot,int(r[iMDD]))
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
def c_feat(subj,mode):
    qs=[3,7] if mode=='anchor' else list(range(1,13))
    parts=[c_q(subj,q) for q in qs]
    if any(p is None for p in parts): return None
    return statfeat(np.vstack(parts))

def eval_set(feats,ys,yb):
    feats,ys,yb=np.array(feats),np.array(ys),np.array(yb)
    aucs,cccs=[],[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(yb),float)
        for tr,te in skf.split(feats,yb):
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(feats[tr],yb[tr]); pb[te]=clf.decision_function(feats[te])
        aucs.append(roc_auc_score(yb,pb))
        kf=KFold(5,shuffle=True,random_state=sd); pr=np.zeros(len(ys),float)
        for tr,te in kf.split(feats):
            reg=make_pipeline(StandardScaler(),Ridge(alpha=10)); reg.fit(feats[tr],ys[tr]); pr[te]=reg.predict(feats[te])
        cccs.append(ccc(ys,pr))
    return np.array(aucs),np.array(cccs)

def collect(dataset,mode):
    F,ys,yb=[],[],[]
    if dataset=='DAIC':
        for pid,(sc,bn) in d_lab().items():
            f=d_feat(pid,mode)
            if f is not None: F.append(f); ys.append(sc); yb.append(bn)
    else:
        for subj,(tot,mdd) in c_lab.items():
            f=c_feat(subj,mode)
            if f is not None: F.append(f); ys.append(tot); yb.append(mdd)
    return F,ys,yb

print('=== 앵커 효과 정식 증명 (10 seed, AUC/CCC) ===',flush=True)
print(f"{'데이터':>6} {'조건':>8} {'AUC':>13} {'CCC':>13}",flush=True)
print('-'*46,flush=True)
results={}
for ds in ['DAIC','CMDC']:
    for mode in (['anchor','all'] if ds=='CMDC' else ['pos','all']):
        F,ys,yb=collect(ds,mode)
        au,cc=eval_set(F,ys,yb)
        results[(ds,mode)]=(au,cc)
        tag='앵커' if mode in('anchor','pos') else '전체'
        print(f"{ds:>6} {tag:>8} {au.mean():.3f}±{au.std():.3f} {cc.mean():.3f}±{cc.std():.3f}  n={len(yb)}",flush=True)

print('\n=== 앵커 vs 전체 쌍체 통계 (Wilcoxon, 10 seed AUC) ===',flush=True)
for ds,am in [('DAIC','pos'),('CMDC','anchor')]:
    a_anchor=results[(ds,am)][0]; a_all=results[(ds,'all')][0]
    try:
        st,p=wilcoxon(a_anchor,a_all)
        d=a_anchor.mean()-a_all.mean()
        print(f'  {ds}: 앵커-전체 ΔAUC={d:+.3f}, Wilcoxon p={p:.4f} {"유의**" if p<0.05 else ""}',flush=True)
    except Exception as e: print(f'  {ds}: {e}',flush=True)
