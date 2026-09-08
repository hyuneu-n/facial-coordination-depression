"""
exp63 — 선행(DSEM tension 통찰) + 우리 twist(mean-augmented SPD coupling).
DSEM이 알려준 것: coupling이 버린 '평균 표정레벨(tension)'이 정보다(exp62 LMVD).
우리 방법: 평균(tension)+공분산(coupling)을 하나의 (d+1)×(d+1) second-moment SPD 행렬로
 통합 → Riemannian tangent. = "coupling이 버린 평균을 기하적으로 되살린 통합 coupling".
 second-moment M=[[C+μμᵀ, μ],[μᵀ,1]] (population-표준화 AU에서 → subject 평균편차=tension 보존).
질문: augmented가 plain coupling을 ≥2코퍼스(특히 LMVD)서 넘나? + 성분 분해(tension이 핵심?).
결과: results/exp63_augmented.csv
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
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
def coupling_cov(AU):  # baseline: per-subject z-score(평균 버림) → cov
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def aug_spd(AUps):  # AUps: population-표준화된 프레임 (평균편차 보존) → second-moment SPD
    d=AUps.shape[1]; mu=AUps.mean(0); c,_=ledoit_wolf(AUps); E=c+np.outer(mu,mu)
    M=np.zeros((d+1,d+1)); M[:d,:d]=E; M[:d,d]=mu; M[d,:d]=mu; M[d,d]=1.0
    return M+1e-4*np.eye(d+1)
def cv_tan(mats,y):
    mats=np.array(mats);y=np.array(y);a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(mats,y):
            X=TangentSpace(metric='riemann').fit(mats[tr]).transform(mats)
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def cv_vec(F,y):
    F=np.nan_to_num(np.array(F));y=np.array(y);a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            sc=StandardScaler().fit(F[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(F[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(F[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def analyze(AUs,y,name):
    y=np.array(y)
    allf=np.vstack(AUs); pm=allf.mean(0); ps=allf.std(0)+1e-6  # population 통계(코퍼스, 무라벨)
    COV=[coupling_cov(au) for au in AUs]
    AUG=[aug_spd((au-pm)/ps) for au in AUs]
    TEN=[((au-pm)/ps).mean(0) for au in AUs]  # tension 벡터(평균편차)
    rc=cv_tan(COV,y); ra=cv_tan(AUG,y); rt=cv_vec(TEN,y)
    print(f'\n===== {name} (n={len(y)}, 우울{y.sum()}) =====',flush=True)
    print(f'  coupling(baseline)      AUC={rc[0]:.3f}±{rc[1]:.3f}',flush=True)
    print(f'  tension만(평균)         AUC={rt[0]:.3f}±{rt[1]:.3f}',flush=True)
    print(f'  ★augmented SPD(우리)    AUC={ra[0]:.3f}±{ra[1]:.3f}  Δ vs coupling {ra[0]-rc[0]:+.3f} {"★넘음" if ra[0]>rc[0]+0.02 else ""}',flush=True)
    return [name,len(y),rc[0],rt[0],ra[0]]
out=[]
# CMDC
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
A=[];Y=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: A.append(np.vstack(parts));Y.append(l)
out.append(analyze(A,Y,'CMDC'))
# DAIC
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
def dau(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None
    h=[x.strip() for x in open(p).readline().split(',')];oi=h.index('success');ai=[h.index(c) for c in AUd]
    fe=[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if fe else None
A=[];Y=[]
for pid,l in dl.items():
    au=dau(pid)
    if au is not None and len(au)>=60: A.append(au);Y.append(l)
out.append(analyze(A,Y,'DAIC'))
# E-DAIC
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
def eau(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*.csv'))
    if not fs:return None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[]
    for ln in open(fs[0]).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if len(fe)>=60 else None
A=[];Y=[]
for pid,l in el.items():
    au=eau(pid)
    if au is not None: A.append(au);Y.append(l)
out.append(analyze(A,Y,'E-DAIC'))
# LMVD
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423): return 1
    if (602<=i<=1116) or (1425<=i<=1824): return 0
    return None
A=[];Y=[]
for f in sorted(glob.glob(str(V/'*.csv'))):
    l=lab(int(Path(f).stem))
    if l is None:continue
    h=[x.strip() for x in open(f).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:continue
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    if len(fe)>=60: A.append(np.nan_to_num(np.array(fe)));Y.append(l)
out.append(analyze(A,Y,'LMVD'))
print('\n판정: augmented SPD가 plain coupling을 ≥2코퍼스(특히 LMVD 대형) +0.02↑ → 비선점 novelty(coupling+tension 기하통합).',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp63_augmented.csv','w') as f:
    f.write('corpus,n,coupling,tension_only,augmented\n')
    for r in [x for x in out if x]:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp63_augmented.csv',flush=True)
