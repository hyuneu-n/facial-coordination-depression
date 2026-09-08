"""
exp71 — 재프레임: 우울 얼굴표정 '표현형(phenotype)' 발견 (detection 아님).
가설: 우울 얼굴신호 = 최소 2축 — 위축(withdrawal, 활성↓) + 초조(agitation, fear-AU↑) —
 이고 독립적이며 서로 다른 증상과 연결. 이질성=발견대상.
검증: (1) 두 축 독립성(상관 낮음?) (2) 우울군 아형 클러스터 안정성 (3) 축↔PHQ항목 연결.
CMDC/DAIC/E-DAIC (per-item PHQ 있는 곳). 결과: results/exp71_phenotype.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
FEAR=['AU01_r','AU02_r','AU04_r','AU05_r','AU20_r']  # 초조/공포(Sci Rep)
def feats(AU, idx):
    """위축=전체 평균활성(음수화), 초조=fear-AU 평균, rigidity=AR1"""
    withdraw = AU.mean(0).mean()  # 전체 활성(낮을수록 위축)
    fi=[idx[a] for a in FEAR if a in idx]
    agit = AU[:,fi].mean() if fi else 0.0
    # rigidity
    ars=[]
    for c in range(AU.shape[1]):
        x=AU[:,c];s=x.std()
        if s>1e-6 and len(x)>10:
            z=(x-x.mean())/s;a=np.polyfit(z[:-1],z[1:],1)[0];ars.append(a)
    return withdraw, agit, (np.mean(ars) if ars else 0)
def analyze(subjAU, phq_items, item_names, y, AUlist, name):
    idx={a:i for i,a in enumerate(AUlist)}
    F=np.array([feats(au,idx) for au in subjAU])  # [withdraw, agit, rigid]
    y=np.array(y)
    # 표준화(코퍼스)
    Fz=(F-F.mean(0))/(F.std(0)+1e-6)
    print(f'\n===== {name} (n={len(y)}, 우울{int(sum(y))}) =====',flush=True)
    # (1) 두 축 독립성 (전체 + 우울군)
    r_all,p_all=spearmanr(Fz[:,0],Fz[:,1])
    dep=y==1
    r_dep,p_dep=spearmanr(Fz[dep,0],Fz[dep,1]) if dep.sum()>5 else (np.nan,1)
    print(f'  (1) 위축축 vs 초조축 상관: 전체 r={r_all:+.3f}(p={p_all:.3f}) | 우울군 r={r_dep:+.3f}(p={p_dep:.3f})  {"→독립적(2축)" if abs(r_dep)<0.4 else "→중복"}',flush=True)
    # (2) 우울군 아형 클러스터 (위축·초조·경직 2D/3D)
    if dep.sum()>=12:
        Xd=Fz[dep][:,:3]
        for k in [2,3]:
            km=KMeans(k,n_init=10,random_state=0).fit(Xd);sil=silhouette_score(Xd,km.labels_)
            sizes=np.bincount(km.labels_)
            print(f'  (2) 우울군 k={k} 아형: silhouette={sil:.3f} 크기={list(sizes)}',flush=True)
    # (3) 축 ↔ PHQ 항목 연결
    if phq_items is not None:
        P=np.array(phq_items,float)
        print(f'  (3) 축↔증상 상관(Spearman):',flush=True)
        for ax,axn in [(0,'위축'),(1,'초조')]:
            cors=[(item_names[j],spearmanr(Fz[:,ax],P[:,j])[0],spearmanr(Fz[:,ax],P[:,j])[1]) for j in range(P.shape[1])]
            cors=[c for c in cors if not np.isnan(c[1])]
            top=sorted(cors,key=lambda c:-abs(c[1]))[:3]
            print(f'     {axn}축 top: '+' | '.join(f'{n}:{r:+.2f}(p{p:.2f})' for n,r,p in top),flush=True)
    return [name,len(y),r_all,r_dep]
out=[]
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
# CMDC (per-item PHQ-1..9, MDD label)
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID=hd.index('ID');iMDD=hd.index('MDD');iP=[hd.index(f'PHQ-{k}') for k in range(1,10)]
info={}
for r in rows[1:]:
    if r[iID] is None:continue
    vals=[r[k] for k in iP]
    if any(v is None or not isinstance(v,(int,float)) for v in vals):continue
    info[str(r[iID]).strip()]=(int(r[iMDD]),[float(v) for v in vals])
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
A=[];Y=[];PH=[]
for s,(l,ph) in info.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: A.append(np.vstack(parts));Y.append(l);PH.append(ph)
out.append(analyze(A,PH,[f'PHQ{k}' for k in range(1,10)],Y,AUc,'CMDC'))
# E-DAIC (Detailed per-item + binary)
E=B/'E-DAIC';el={};det={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
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
A=[];Y=[];PH=[]
for pid,l in el.items():
    if pid not in det:continue
    au=eau(pid)
    if au is not None: A.append(au);Y.append(l);PH.append(det[pid])
short=['무쾌감','기분','수면','피로','식욕','실패','집중','정신운동']
out.append(analyze(A,PH,short,Y,AUc,'E-DAIC'))
print('\n판정: 위축·초조 두 축이 독립(우울군서 |r|<0.4) + 서로 다른 증상과 연결 + 아형 silhouette>0.3 →',flush=True)
print('      "우울 얼굴 표현형(위축형/초조형)" 발견 = detection 넘는 discovery novelty.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp71_phenotype.csv','w') as f:
    f.write('corpus,n,axis_corr_all,axis_corr_dep\n')
    for r in [x for x in out if x]:f.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp71_phenotype.csv',flush=True)
