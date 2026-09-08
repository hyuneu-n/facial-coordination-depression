"""
exp72 — consolidate: '해석가능 얼굴지표 × PHQ증상' 상관 매트릭스, cross-corpus 재현.
7개 얼굴지표 × 8개 PHQ항목. CMDC+E-DAIC 둘 다 같은 부호 & p<0.05 인 연결만 = 재현된 발견.
이론가설: 긍정표정(AU6/12)↓↔무쾌감 / 경직(rigidity)↔정신운동 / fear↔?
결과: results/exp72_symptom_map.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
# 얼굴지표 정의 (AU_r 이름)
IDX_DEF={
 'expressivity_all': None,  # 전체 평균(특수처리)
 'positive_smile': ['AU06_r','AU12_r'],
 'fear_tension': ['AU01_r','AU02_r','AU04_r','AU05_r','AU20_r'],
 'sadness': ['AU01_r','AU04_r','AU15_r'],
 'mouth': ['AU25_r','AU26_r'],
}
SYMPT=['무쾌감','기분','수면','피로','식욕','실패','집중','정신운동']
def subj_indices(AU, colidx):
    means=AU.mean(0)
    out={}
    out['expressivity_all']=means.mean()
    for k,aus in IDX_DEF.items():
        if aus is None:continue
        ii=[colidx[a] for a in aus if a in colidx]
        out[k]=means[ii].mean() if ii else np.nan
    # rigidity(AR1 평균), variability(std 평균)
    ars=[]
    for c in range(AU.shape[1]):
        x=AU[:,c];s=x.std()
        if s>1e-6 and len(x)>10:
            z=(x-x.mean())/s;ars.append(np.polyfit(z[:-1],z[1:],1)[0])
    out['rigidity']=np.mean(ars) if ars else np.nan
    out['variability']=AU.std(0).mean()
    return out
INDEX_NAMES=['expressivity_all','positive_smile','fear_tension','sadness','mouth','rigidity','variability']
def build(subjAU, phq, colidx):
    F={k:[] for k in INDEX_NAMES}; P=[]
    for au,ph in zip(subjAU,phq):
        d=subj_indices(au,colidx)
        for k in INDEX_NAMES: F[k].append(d.get(k,np.nan))
        P.append(ph)
    F={k:np.array(v) for k,v in F.items()}; P=np.array(P,float)
    return F,P
def corr_matrix(F,P,name):
    print(f'\n===== {name} (n={len(P)}) 얼굴지표×증상 Spearman r (p<.05만 표시) =====',flush=True)
    M={}
    for ix in INDEX_NAMES:
        row={}
        for j,sy in enumerate(SYMPT):
            r,p=spearmanr(F[ix],P[:,j],nan_policy='omit')
            row[sy]=(r,p)
        M[ix]=row
        sig=[f'{sy}:{r:+.2f}' for sy,(r,p) in row.items() if p<0.05]
        if sig: print(f'  {ix:16s} {" | ".join(sig)}',flush=True)
    return M
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
# CMDC (PHQ-1..8)
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID=hd.index('ID');iP=[hd.index(f'PHQ-{k}') for k in range(1,9)]
info={}
for r in rows[1:]:
    if r[iID] is None:continue
    vals=[r[k] for k in iP]
    if any(v is None or not isinstance(v,(int,float)) for v in vals):continue
    info[str(r[iID]).strip()]=[float(v) for v in vals]
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
A=[];PH=[]
for s,ph in info.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: A.append(np.vstack(parts));PH.append(ph)
Fc,Pc=build(A,PH,{a:i for i,a in enumerate(AUc)})
Mc=corr_matrix(Fc,Pc,'CMDC')
# E-DAIC (Detailed 8)
E=B/'E-DAIC';det={}
cols=['PHQ_8NoInterest','PHQ_8Depressed','PHQ_8Sleep','PHQ_8Tired','PHQ_8Appetite','PHQ_8Failure','PHQ_8Concentrating','PHQ_8Moving']
for r in csv.DictReader(open(E/'labels'/'Detailed_PHQ8_Labels.csv')):
    try:det[r['Participant_ID'].strip()]=[float(r[c]) for c in cols]
    except:pass
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
A=[];PH=[]
for pid,ph in det.items():
    au=eau(pid)
    if au is not None: A.append(au);PH.append(ph)
Fe,Pe=build(A,PH,{a:i for i,a in enumerate(AUc)})
Me=corr_matrix(Fe,Pe,'E-DAIC')
# 재현된 연결: 양쪽 p<0.05 & 같은 부호
print('\n===== ★재현된 얼굴지표↔증상 연결 (CMDC & E-DAIC 둘 다 p<.05, 동일부호) =====',flush=True)
rep=[]
for ix in INDEX_NAMES:
    for sy in SYMPT:
        rc,pc=Mc[ix][sy]; re,pe=Me[ix][sy]
        if pc<0.05 and pe<0.05 and np.sign(rc)==np.sign(re):
            print(f'  ★ {ix} ↔ {sy}: CMDC r={rc:+.2f} | E-DAIC r={re:+.2f}',flush=True)
            rep.append((ix,sy,rc,re))
if not rep: print('  (양쪽 동시 유의 연결 없음 — 재현성 약함)',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp72_symptom_map.csv','w') as f:
    f.write('index,symptom,cmdc_r,cmdc_p,edaic_r,edaic_p\n')
    for ix in INDEX_NAMES:
        for sy in SYMPT:
            rc,pc=Mc[ix][sy];re,pe=Me[ix][sy]
            f.write(f'{ix},{sy},{rc:.3f},{pc:.3f},{re:.3f},{pe:.3f}\n')
print('DONE → exp72_symptom_map.csv',flush=True)
