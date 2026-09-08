"""
exp57 — idea A: 감정 회복 동역학 (affective recovery dynamics).
within-session 감정 perturbation(질문) 후 표정 valence의 관성/회복을 측정.
 valence(t)=z(AU6+AU12 긍정)-z(AU4+AU15+AU1 부정).
 특징: inertia(AR1, Kuppens 정서관성을 얼굴에 이식), recovery_time(부정 trough 후 baseline 복귀),
       val_mean(잔류 부정), val_std(blunting), neg_rate.
핵심질문: 이 동역학이 정적 coupling에 '정보를 더하나'(combined>coupling)? + inertia MDD>HC?
CMDC/DAIC/LMVD. 결과: results/exp57_affective.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10
POS=['AU06_r','AU12_r']; NEG=['AU04_r','AU15_r','AU01_r']
def valence(AU,idx):
    def z(x):return (x-x.mean())/(x.std()+1e-6)
    p=sum(AU[:,idx[a]] for a in POS if a in idx); n=sum(AU[:,idx[a]] for a in NEG if a in idx)
    return z(p)-z(n)
def ar1(v):
    v=v-v.mean(); s=v.std()
    if s<1e-8 or len(v)<10: return np.nan
    a,b=v[1:],v[:-1]; d=a.std()*b.std()
    return float(((a-a.mean())*(b-b.mean())).mean()/d) if d>1e-12 else np.nan
def recovery(v):
    """부정 trough 후 baseline(median) 복귀까지 프레임수(관성=느린회복)"""
    base=np.median(v); mad=np.median(np.abs(v-base))+1e-6
    tr,_=find_peaks(-v, prominence=1.0*mad, distance=15); times=[]
    for t0 in tr:
        if v[t0]>=base: continue
        for t in range(t0+1,min(t0+150,len(v))):
            if v[t]>=base: times.append(t-t0); break
    return np.mean(times) if times else np.nan, len(tr)/len(v)
def affect_feats(segs):
    """segs: valence 신호 리스트(질문별 or 전체). 특징 5개."""
    ars=[];recs=[];rate=[];allv=[]
    for v in segs:
        if len(v)<20: continue
        a=ar1(v)
        if not np.isnan(a): ars.append(a)
        rt,rr=recovery(v)
        if not np.isnan(rt): recs.append(rt)
        rate.append(rr); allv.append(v)
    if not allv: return None
    allc=np.concatenate(allv)
    return np.array([np.mean(ars) if ars else 0, np.mean(recs) if recs else 75,
                     np.mean(rate) if rate else 0, allc.mean(), allc.std()])
FN=['inertia_AR1','recovery_time','neg_rate','val_mean','val_std']
def cov_of(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def cv(COV,F,y,mode):
    COV=np.array(COV);F=np.nan_to_num(np.array(F));y=np.array(y);a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            if mode=='aff': X=F
            else:
                T=TangentSpace(metric='riemann').fit(COV[tr]);Xt=T.transform(COV)
                X=Xt if mode=='cov' else np.column_stack([Xt,F])
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def analyze(subj_segAU, subj_wholeAU, y, idx, name):
    y=np.array(y); F=[];COV=[]
    for segs_au, whole in zip(subj_segAU, subj_wholeAU):
        vs=[valence(a,idx) for a in segs_au if len(a)>=20]
        f=affect_feats(vs)
        F.append(f if f is not None else np.zeros(5)); COV.append(cov_of(whole))
    F=np.array(F)
    print(f'\n===== {name} (n={len(y)}, 우울{y.sum()}) =====',flush=True)
    for j,nm in enumerate(FN):
        md,hc=np.nanmean(F[y==1,j]),np.nanmean(F[y==0,j])
        try:_,p=mannwhitneyu(F[y==1,j][~np.isnan(F[y==1,j])],F[y==0,j][~np.isnan(F[y==0,j])])
        except:p=1
        star=' ★' if nm=='inertia_AR1' else ''
        print(f'  {nm:13s} MDD={md:.3f} HC={hc:.3f} ({"↑" if md>hc else "↓"}우울) p={p:.3f}{star}',flush=True)
    ra=cv(COV,F,y,'aff');rc=cv(COV,F,y,'cov');rb=cv(COV,F,y,'comb')
    print(f'  affect동역학 단독 AUC={ra[0]:.3f}±{ra[1]:.3f}',flush=True)
    print(f'  정적 coupling  AUC={rc[0]:.3f}±{rc[1]:.3f}',flush=True)
    print(f'  결합           AUC={rb[0]:.3f}±{rb[1]:.3f}  {"★coupling에 정보추가" if rb[0]>rc[0]+0.02 else "추가 없음"}',flush=True)
    return [name,len(y),ra[0],rc[0],rb[0]]
out=[]
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
# CMDC: 질문별 세그먼트 그대로 활용(=within-session perturbation 단위)
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
idxC={a:i for i,a in enumerate(AUc)}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
segAU=[];wholeAU=[];Y=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: segAU.append(parts);wholeAU.append(np.vstack(parts));Y.append(l)
out.append(analyze(segAU,wholeAU,Y,idxC,'CMDC'))
# DAIC: 전체(질문 세그먼트 태깅 없어 whole valence)
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
idxD={a:i for i,a in enumerate(AUd)}
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
segAU=[];wholeAU=[];Y=[]
for pid,l in dl.items():
    au=dau(pid)
    if au is not None and len(au)>=60:
        # 슬라이딩 세그먼트(질문태깅 없어 300프레임 창)로 perturbation 단위 근사
        segs=[au[i:i+300] for i in range(0,len(au),300) if len(au[i:i+300])>=60]
        segAU.append(segs);wholeAU.append(au);Y.append(l)
out.append(analyze(segAU,wholeAU,Y,idxD,'DAIC'))
# LMVD
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423): return 1
    if (602<=i<=1116) or (1425<=i<=1824): return 0
    return None
segAU=[];wholeAU=[];Y=[]
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
    if len(fe)>=60:
        au=np.array(fe);segs=[au[i:i+300] for i in range(0,len(au),300) if len(au[i:i+300])>=60]
        segAU.append(segs);wholeAU.append(au);Y.append(l)
out.append(analyze(segAU,wholeAU,Y,idxC,'LMVD'))
print('\n판정: (1)inertia MDD>HC 유의 + (2)결합>coupling(+0.02↑) 이 ≥2코퍼스 일관 →',flush=True)
print('      "얼굴 정서관성/회복지연"=coupling에 없는 goal-정합 새 신호(novelty 후보).',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp57_affective.csv','w') as f:
    f.write('corpus,n,affect_AUC,coupling_AUC,combined_AUC\n')
    for r in out:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp57_affective.csv',flush=True)
