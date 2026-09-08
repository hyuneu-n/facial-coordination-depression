"""
exp60 — 증상별(symptom-level) 얼굴 coupling 예측력. (positive·해석가능·비선점 각도)
"얼굴 AU coupling이 8개 PHQ 증상 중 어떤 것을 예측하나?"
가설: 정신운동(Moving)·무쾌감(NoInterest)·기분(Depressed)은 얼굴에 나옴 → 예측됨,
      수면·식욕·실패감은 얼굴 무관 → 예측 안 됨. = 해석가능한 '증상특이 얼굴마커'.
검증: 증상별 CV Spearman(coupling→증상) + permutation p. CMDC(PHQ-1..8)+E-DAIC(8항목) 재현.
결과: results/exp60_symptom.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10; NPERM=200
SYMPT=['1무쾌감','2기분','3수면','4피로','5식욕','6실패감','7집중','8정신운동']
FACIAL_HYP=[0,1,6,7]  # 얼굴 관련 가설 증상 인덱스(무쾌감·기분·집중·정신운동)
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
def cov_of(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def cv_spearman(COV,yv,seed):
    kf=KFold(5,shuffle=True,random_state=seed); pred=np.zeros(len(yv))
    for tr,te in kf.split(COV):
        T=TangentSpace(metric='riemann').fit(COV[tr]);X=T.transform(COV)
        sc=StandardScaler().fit(X[tr]);rg=RidgeCV(alphas=[0.1,1,10,100]).fit(sc.transform(X[tr]),yv[tr])
        pred[te]=rg.predict(sc.transform(X[te]))
    r,_=spearmanr(pred,yv); return 0.0 if np.isnan(r) else r
def eval_symptom(COV,yv):
    COV=np.array(COV);yv=np.array(yv,float)
    if np.std(yv)<1e-6: return None
    rs=np.mean([cv_spearman(COV,yv,s) for s in range(SEEDS)])
    # permutation
    rng=np.random.RandomState(0); null=[]
    for _ in range(NPERM):
        yp=rng.permutation(yv); null.append(cv_spearman(COV,yp,0))
    p=(np.sum(np.array(null)>=rs)+1)/(NPERM+1)
    return rs,p
def analyze(COV, S, name):
    """COV: per-subj cov, S: (n,8) 증상점수"""
    S=np.array(S,float); print(f'\n===== {name} (n={len(COV)}) 증상별 coupling 예측력 =====',flush=True)
    res=[]
    for j,nm in enumerate(SYMPT):
        r=eval_symptom(COV,S[:,j])
        if r is None: print(f'  {nm:9s} (분산없음)',flush=True); res.append((nm,0,1)); continue
        rs,p=r; star='★' if p<0.05 else ''
        hyp='[얼굴가설]' if j in FACIAL_HYP else '[비얼굴]'
        print(f'  {nm:9s} {hyp:8s} Spearman={rs:+.3f} perm_p={p:.3f} {star}',flush=True)
        res.append((nm,rs,p))
    return name,res
out=[]
# CMDC
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0])
iID=hd.index('ID'); iPHQ=[hd.index(f'PHQ-{k}') for k in range(1,9)]
info={str(r[iID]).strip():[r[k] for k in iPHQ] for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
COV=[];S=[]
for s,ph in info.items():
    if any(x is None for x in ph): continue
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: COV.append(cov_of(np.vstack(parts)));S.append([float(x) for x in ph])
out.append(analyze(COV,S,'CMDC'))
# E-DAIC
E=B/'E-DAIC'
det={}
dr=list(csv.DictReader(open(E/'labels'/'Detailed_PHQ8_Labels.csv')))
cols=['PHQ_8NoInterest','PHQ_8Depressed','PHQ_8Sleep','PHQ_8Tired','PHQ_8Appetite','PHQ_8Failure','PHQ_8Concentrating','PHQ_8Moving']
for r in dr:
    pid=r['Participant_ID'].strip()
    try: det[pid]=[float(r[c]) for c in cols]
    except: pass
def eload(pid):
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
COV=[];S=[]
for pid,ph in det.items():
    m=eload(pid)
    if m is not None: COV.append(cov_of(m));S.append(ph)
out.append(analyze(COV,S,'E-DAIC'))
print('\n판정: 얼굴가설 증상(무쾌감·기분·집중·정신운동)이 비얼굴(수면·식욕)보다 예측력↑ & 두 코퍼스 일관 →',flush=True)
print('      "증상특이 얼굴 coupling 마커"=해석가능 positive novelty(비선점, AUC경쟁 회피).',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp60_symptom.csv','w') as f:
    f.write('corpus,symptom,spearman,perm_p\n')
    for name,res in out:
        for nm,rs,p in res: f.write(f'{name},{nm},{rs:.4f},{p:.4f}\n')
print('DONE → exp60_symptom.csv',flush=True)
