"""
exp70 — ★목표 정합 재프레임★: subthreshold(전조/prodromal) 검출 + severity 밴드별 signature.
Sci Rep 2025: subthreshold depression(경계)=긍정표정↓+공포/긴장 AU(1/5/20)↑, 전조 마커.
내 과오: binary(중증 vs 정상)만 봄 → PHQ<10 '정상'에 subthreshold(5-9) 섞임=전조를 묻음.
연속 PHQ로 밴드: H(0-4)/StD(5-9,전조)/Mod+(10+).
 TaskA: StD vs H (전조 검출, 어려움).  TaskB: 밴드별 expressivity/tension/공포AU 추세(전조 signature?).
CMDC/DAIC/E-DAIC. 결과: results/exp70_subthreshold.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10
FEAR=['AU01_r','AU02_r','AU04_r','AU05_r','AU20_r']  # Sci Rep 공포/긴장 AU
def cov_of(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def cvauc(COV,EXP,y,mode):
    COV=np.array(COV);EXP=np.nan_to_num(np.array(EXP));y=np.array(y)
    if len(set(y))<2 or min(np.bincount(y))<5: return float('nan')
    a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(COV,y):
            if mode=='exp':X=EXP
            else:
                T=TangentSpace(metric='riemann').fit(COV[tr]).transform(COV);X=T if mode=='cov' else np.column_stack([T,EXP])
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a)
def band(p):
    if p<=4:return 0   # healthy
    if p<=9:return 1   # subthreshold(전조)
    return 2           # moderate+
def analyze(subjAU, phq, AUlist, name):
    idx={a:i for i,a in enumerate(AUlist)}
    fear_i=[idx[a] for a in FEAR if a in idx]
    COV=[cov_of(au) for au in subjAU]
    means=np.array([au.mean(0) for au in subjAU]); pm=means.mean(0);ps=means.std(0)+1e-6
    TEN=(means-pm)/ps  # 전체 tension
    expr_all=TEN  # expressivity 벡터
    phq=np.array(phq); bands=np.array([band(p) for p in phq])
    print(f'\n===== {name} (n={len(phq)}) H={np.sum(bands==0)} StD={np.sum(bands==1)} Mod+={np.sum(bands==2)} =====',flush=True)
    # TaskB: 밴드별 평균 활성(tension 합) + 공포AU
    ten_scalar=means.mean(1); fear_scalar=means[:,fear_i].mean(1) if fear_i else np.zeros(len(phq))
    for b,nm in [(0,'H'),(1,'StD전조'),(2,'Mod+')]:
        m=bands==b
        if m.sum()>0: print(f'  {nm:8s} 평균활성={ten_scalar[m].mean():.3f} 공포AU={fear_scalar[m].mean():.3f}',flush=True)
    # dose-response: PHQ vs 활성/공포
    rt,pt=spearmanr(phq,ten_scalar); rf,pf=spearmanr(phq,fear_scalar)
    print(f'  PHQ상관: 평균활성 r={rt:+.3f}(p={pt:.3f}) | 공포AU r={rf:+.3f}(p={pf:.3f})',flush=True)
    # TaskA: StD vs H (전조 검출)
    mA=(bands==0)|(bands==1); yA=(bands[mA]==1).astype(int)
    COVa=[COV[i] for i in range(len(COV)) if mA[i]]; EXa=expr_all[mA]
    a_cpl=cvauc(COVa,EXa,yA,'cov'); a_cmb=cvauc(COVa,EXa,yA,'comb')
    # 참조: Mod+ vs H (쉬운 표준 task)
    mB=(bands==0)|(bands==2); yB=(bands[mB]==2).astype(int)
    COVb=[COV[i] for i in range(len(COV)) if mB[i]]; EXb=expr_all[mB]
    b_cpl=cvauc(COVb,EXb,yB,'cov'); b_cmb=cvauc(COVb,EXb,yB,'comb')
    print(f'  ★TaskA 전조검출 StD vs H:  coupling={a_cpl:.3f} | +expr={a_cmb:.3f}',flush=True)
    print(f'   TaskB 표준 Mod+ vs H:     coupling={b_cpl:.3f} | +expr={b_cmb:.3f}',flush=True)
    return [name,int((bands==0).sum()),int((bands==1).sum()),int((bands==2).sum()),rt,rf,a_cpl,a_cmb,b_cpl,b_cmb]
out=[]
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
# CMDC (PHQtotal)
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID=hd.index('ID');iPHQ=[hd.index(f'PHQ-{k}') for k in range(1,10)]
pq={}
for r in rows[1:]:
    if r[iID] is None: continue
    vals=[r[k] for k in iPHQ]
    if any(v is None or not isinstance(v,(int,float)) for v in vals): continue
    pq[str(r[iID]).strip()]=float(sum(vals))
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
A=[];P=[]
for s,pv in pq.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: A.append(np.vstack(parts));P.append(float(pv))
out.append(analyze(A,P,AUc,'CMDC'))
# DAIC (PHQ8_Score)
D=B/'DAIC_WOZ';dscore={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            k=[x for x in r if 'PHQ' in x and 'Score' in x]
            if k: dscore[r['Participant_ID'].strip()]=float(r[k[0]])
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
A=[];P=[]
for pid,pv in dscore.items():
    au=dau(pid)
    if au is not None and len(au)>=60: A.append(au);P.append(pv)
out.append(analyze(A,P,AUd,'DAIC'))
# E-DAIC (PHQ_Score)
E=B/'E-DAIC';escore={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            k=[x for x in r if 'PHQ' in x and 'Score' in x]
            if k and r[k[0]] not in(None,''): escore[r['Participant_ID'].strip()]=float(r[k[0]])
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
A=[];P=[]
for pid,pv in escore.items():
    au=eau(pid)
    if au is not None: A.append(au);P.append(pv)
out.append(analyze(A,P,AUc,'E-DAIC'))
print('\n판정: 전조검출(StD vs H) AUC가 chance 넘고, 밴드별 signature(전조=공포↑? 중증=활성↓?) 나오면',flush=True)
print('      = 목표(전조/조기) 정합 + Sci Rep 근거 + 기존데이터 가능 = 진짜 각도.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp70_subthreshold.csv','w') as f:
    f.write('corpus,nH,nStD,nMod,r_activity,r_fear,StDvsH_cpl,StDvsH_comb,ModvsH_cpl,ModvsH_comb\n')
    for r in [x for x in out if x]:f.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp70_subthreshold.csv',flush=True)
