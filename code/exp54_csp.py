"""
exp54 — CSP식 판별 spatial-covariance 필터를 얼굴 AU coupling에 이식 (경로 A).
교수님 Si-MCSP(CSP) 계보 표준 방법을 facial AU 우울판별에 적용.
산출: (1) CSP AUC vs tangent baseline(성능 동등이면 OK), (2) 해석가능 판별 필터(AU 패턴),
      (3) 3코퍼스 필터 일관성. novelty=method/framework 이식 + 해석 + 재현(교수님 기준).
결과: results/exp54_csp.csv + exp54_patterns.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import CSP
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10; NF=6
AUc=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45']
AUcR=[a+'_r' for a in AUc]
AUd=['AU01','AU02','AU04','AU05','AU06','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU25','AU26']
AUdR=[a+'_r' for a in AUd]
def cov_mat(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def cv(COV, y, mode):
    COV=np.array(COV); y=np.array(y); a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s); pb=np.zeros(len(y))
        for tr,te in skf.split(COV,y):
            if mode=='tan':
                X=TangentSpace(metric='riemann').fit(COV[tr]).transform(COV)
            else:
                cs=CSP(nfilter=NF,metric='riemann',log=True).fit(COV[tr],y[tr]); X=cs.transform(COV)
            sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=lr.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def analyze(subjAU, y, AUnames, name, prows):
    y=np.array(y); COV=np.array([cov_mat(a) for a in subjAU])
    rt=cv(COV,y,'tan'); rc=cv(COV,y,'csp')
    print(f'\n===== {name} (n={len(y)}, 우울{y.sum()}) =====',flush=True)
    print(f'  tangent(baseline) AUC={rt[0]:.3f}±{rt[1]:.3f}',flush=True)
    print(f'  CSP(판별필터)     AUC={rc[0]:.3f}±{rc[1]:.3f}  ({"동등" if abs(rc[0]-rt[0])<0.03 else ("우위" if rc[0]>rt[0] else "열위")})',flush=True)
    # 전체 fit → 해석 패턴 (가장 판별적인 첫 필터 = 우울↑ 방향)
    cs=CSP(nfilter=NF,metric='riemann',log=True).fit(COV,y)
    P=cs.patterns_  # (NF, nAU): 각 필터의 AU 활성 패턴
    top=P[0]/ (np.abs(P[0]).max()+1e-9)  # 첫 필터 정규화
    order=np.argsort(-np.abs(top))
    print(f'  [해석] 최상위 판별필터 주요 AU (|loading| 상위5):',flush=True)
    for j in order[:5]:
        print(f'      {AUnames[j]:5s} {top[j]:+.2f}',flush=True)
    for j,au in enumerate(AUnames): prows.append([name,au,f'{top[j]:.4f}'])
    return [name,len(y),rt[0],rc[0]]
out=[];prows=[]
# CMDC
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUcR]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
sA=[];sy=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: sA.append(np.vstack(parts));sy.append(l)
out.append(analyze(sA,sy,AUc,'CMDC',prows))
# DAIC
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
def dau(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None
    h=[x.strip() for x in open(p).readline().split(',')];oi=h.index('success');ai=[h.index(c) for c in AUdR]
    fe=[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if fe else None
sA=[];sy=[]
for pid,l in dl.items():
    au=dau(pid)
    if au is not None and len(au)>=60: sA.append(au);sy.append(l)
out.append(analyze(sA,sy,AUd,'DAIC',prows))
# LMVD
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423): return 1
    if (602<=i<=1116) or (1425<=i<=1824): return 0
    return None
sA=[];sy=[]
for f in sorted(glob.glob(str(V/'*.csv'))):
    l=lab(int(Path(f).stem))
    if l is None:continue
    h=[x.strip() for x in open(f).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUcR]
    except:continue
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    if len(fe)>=60: sA.append(np.array(fe));sy.append(l)
out.append(analyze(sA,sy,AUc,'LMVD',prows))
print('\n판정: CSP AUC가 tangent와 동등(±0.03) & 판별필터가 해석가능/3코퍼스 일관 →',flush=True)
print('      "판별 AU-coupling 패턴" 방법이식 논문 성립(성능SOTA 아닌 해석/재현 novelty).',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp54_csp.csv','w') as f:
    f.write('corpus,n,tangent_AUC,CSP_AUC\n')
    for r in out:f.write(','.join(str(x) for x in r)+'\n')
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp54_patterns.csv','w') as f:
    f.write('corpus,AU,top_filter_loading\n')
    for r in prows:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp54_csp.csv + exp54_patterns.csv',flush=True)
