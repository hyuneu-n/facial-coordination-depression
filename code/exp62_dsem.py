"""
exp62 — idea(Psychoneuroendocrinology 2026 DSEM 근거): 표정 dynamic 분해.
스트레스/우울 표정 = rigidity(경직) + tension(긴장) + unpredictability(예측불가).
 DSEM식 per-AU 분해: mean/std(tension=tonic level, ★내가 z-score로 늘 버렸던 것),
   AR1(rigidity), innovation variance(unpredictability=AR 잔차분산).
질문: 이 분해가 coupling에 정보를 더하나? (coupling은 mean/level을 버림) + 기전.
CMDC/DAIC/E-DAIC/LMVD. 결과: results/exp62_dsem.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
def cov_of(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def dsem_feats(AU):
    """per-AU: mean,std(raw=tension) + AR1(rigidity), innovation_var(unpredictability, z-scored)"""
    means=AU.mean(0); stds=AU.std(0)
    ar=[];inno=[]
    for c in range(AU.shape[1]):
        x=AU[:,c]; s=x.std()
        if s<1e-6 or len(x)<10: ar.append(0);inno.append(0);continue
        z=(x-x.mean())/s
        a=np.polyfit(z[:-1],z[1:],1); pred=a[0]*z[:-1]+a[1]; res=z[1:]-pred
        ar.append(a[0]); inno.append(res.var())
    # 집계 통계도(전반 tension/rigidity/unpredictability)
    agg=[means.mean(),stds.mean(),np.mean(ar),np.mean(inno)]
    return np.concatenate([means,stds,ar,inno,agg]), np.mean(ar), means.mean(), np.mean(inno)
def cv(COV,F,y,mode):
    COV=np.array(COV);F=np.nan_to_num(np.array(F));y=np.array(y);a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            if mode=='dsem':
                k=min(20,len(tr)-1,F.shape[1]);pc=PCA(k).fit(F[tr]);X=pc.transform(F)
            else:
                T=TangentSpace(metric='riemann').fit(COV[tr]);Xt=T.transform(COV)
                if mode=='cov':X=Xt
                else:
                    k=min(20,len(tr)-1,F.shape[1]);pc=PCA(k).fit(F[tr]);X=np.column_stack([Xt,pc.transform(F)])
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def analyze(AUs,y,name):
    y=np.array(y);F=[];RIG=[];TEN=[];UNP=[];COV=[]
    for au in AUs:
        f,rg,tn,un=dsem_feats(au);F.append(f);RIG.append(rg);TEN.append(tn);UNP.append(un);COV.append(cov_of(au))
    F=np.array(F);RIG=np.array(RIG);TEN=np.array(TEN);UNP=np.array(UNP)
    print(f'\n===== {name} (n={len(y)}, 우울{y.sum()}) =====',flush=True)
    for nm,v in [('rigidity(AR1)',RIG),('tension(mean)',TEN),('unpredict(innov)',UNP)]:
        md,hc=v[y==1].mean(),v[y==0].mean()
        try:_,p=mannwhitneyu(v[y==1],v[y==0])
        except:p=1
        print(f'  {nm:16s} MDD={md:.4f} HC={hc:.4f} ({"↑" if md>hc else "↓"}우울) p={p:.3f}',flush=True)
    rd=cv(COV,F,y,'dsem');rc=cv(COV,F,y,'cov');rb=cv(COV,F,y,'comb')
    print(f'  DSEM 단독     AUC={rd[0]:.3f}±{rd[1]:.3f}',flush=True)
    print(f'  coupling      AUC={rc[0]:.3f}±{rc[1]:.3f}',flush=True)
    print(f'  결합          AUC={rb[0]:.3f}±{rb[1]:.3f}  {"★추가" if rb[0]>rc[0]+0.02 else "추가 없음"}',flush=True)
    return [name,len(y),rd[0],rc[0],rb[0]]
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
print('\n판정: DSEM(특히 tension/mean·unpredictability)이 coupling에 ≥2코퍼스 +0.02↑ or 기전 일관 → 비선점 후보.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp62_dsem.csv','w') as f:
    f.write('corpus,n,dsem_only,coupling,combined\n')
    for r in [x for x in out if x]:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp62_dsem.csv',flush=True)
