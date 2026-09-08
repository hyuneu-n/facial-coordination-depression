"""
exp53 — 고차/비선형 AU coupling이 pairwise 공분산(선점작 arXiv2407.13753)을 넘나?
선점작·우리 baseline = 전부 2차 pairwise covariance. 아직 아무도 안 쓴:
  (2) MI    : pairwise 상호정보(비선형 의존, 공분산이 못 잡는 nonlinear coupling)
  (3) coskew: 3차 표준화 co-moment E[z_i z_j z_k] (i<j<k) = 고차(triadic) synchrony
비교: cov(baseline) / mi / coskew / cov+coskew / cov+mi. 하나라도 cov 유의미하게 넘고
  ≥2 코퍼스 일관 → positive novelty 후보. 결과: results/exp53_higher_order.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from itertools import combinations
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=5; KPCA=15
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
def cov_mat(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def mi_vec(AU, Bn=6):
    d=AU.shape[1]; edges=[np.quantile(AU[:,i],np.linspace(0,1,Bn+1)) for i in range(d)]
    dig=np.stack([np.clip(np.digitize(AU[:,i],edges[i][1:-1]),0,Bn-1) for i in range(d)],1)
    out=[]
    for i,j in combinations(range(d),2):
        h=np.histogram2d(dig[:,i],dig[:,j],bins=[Bn,Bn])[0]; p=h/h.sum()+1e-12
        pi=p.sum(1,keepdims=True); pj=p.sum(0,keepdims=True)
        out.append(float((p*np.log(p/(pi*pj))).sum()))
    return np.array(out)
def coskew_vec(AU):
    z=(AU-AU.mean(0))/(AU.std(0)+1e-6); d=AU.shape[1]
    return np.array([float((z[:,i]*z[:,j]*z[:,k]).mean()) for i,j,k in combinations(range(d),3)])
def pipe(): return Pipeline([('sc',StandardScaler()),('pca',PCA(KPCA)),('lr',LogisticRegression(max_iter=3000,class_weight='balanced'))])
def cv_vec(F, y):
    F=np.nan_to_num(F); a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s); pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            k=min(KPCA,len(tr)-1,F.shape[1]); pl=Pipeline([('sc',StandardScaler()),('pca',PCA(k)),('lr',LogisticRegression(max_iter=3000,class_weight='balanced'))])
            pl.fit(F[tr],y[tr]); pb[te]=pl.decision_function(F[te])
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def cv_tan(COV, y, extra=None):
    COV=np.array(COV); a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s); pb=np.zeros(len(y))
        for tr,te in skf.split(COV,y):
            X=TangentSpace(metric='riemann').fit(COV[tr]).transform(COV)
            if extra is not None:
                k=min(KPCA,len(tr)-1,extra.shape[1]); pc=PCA(k).fit(extra[tr]); X=np.column_stack([X,pc.transform(extra)])
            sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=lr.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def analyze(subjAU, y, name):
    y=np.array(y); COV=[cov_mat(a) for a in subjAU]
    MI=np.array([mi_vec(a) for a in subjAU]); CS=np.array([coskew_vec(a) for a in subjAU])
    rc=cv_tan(COV,y); rm=cv_vec(MI,y); rs=cv_vec(CS,y)
    rcs=cv_tan(COV,y,CS); rcm=cv_tan(COV,y,MI)
    print(f'\n===== {name} (n={len(y)}, 우울{y.sum()}) =====',flush=True)
    print(f'  cov(baseline) {rc[0]:.3f}±{rc[1]:.3f}',flush=True)
    print(f'  MI(비선형)    {rm[0]:.3f}±{rm[1]:.3f}  {"★>cov" if rm[0]>rc[0]+0.02 else ""}',flush=True)
    print(f'  coskew(3차)   {rs[0]:.3f}±{rs[1]:.3f}  {"★>cov" if rs[0]>rc[0]+0.02 else ""}',flush=True)
    print(f'  cov+coskew    {rcs[0]:.3f}±{rcs[1]:.3f}  {"★>cov" if rcs[0]>rc[0]+0.02 else ""}',flush=True)
    print(f'  cov+MI        {rcm[0]:.3f}±{rcm[1]:.3f}  {"★>cov" if rcm[0]>rc[0]+0.02 else ""}',flush=True)
    return [name,len(y),rc[0],rm[0],rs[0],rcs[0],rcm[0]]
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
sA=[];sy=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: sA.append(np.vstack(parts));sy.append(l)
out.append(analyze(sA,sy,'CMDC'))
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
sA=[];sy=[]
for pid,l in dl.items():
    au=dau(pid)
    if au is not None and len(au)>=60: sA.append(au);sy.append(l)
out.append(analyze(sA,sy,'DAIC'))
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
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:continue
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    if len(fe)>=60: sA.append(np.array(fe));sy.append(l)
out.append(analyze(sA,sy,'LMVD'))
print('\n판정: MI/coskew/결합 중 하나가 cov를 ≥2코퍼스서 +0.02↑ 일관 초과 → positive novelty 후보.',flush=True)
print('      다 못 넘으면 → 우울 얼굴신호는 2차 pairwise로 이미 포화(고차 정보이득 없음) 확정.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp53_higher_order.csv','w') as f:
    f.write('corpus,n,cov,MI,coskew,cov_coskew,cov_MI\n')
    for r in out:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp53_higher_order.csv',flush=True)
