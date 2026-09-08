"""
exp66 — 이득 통계 엄밀성: 'coupling+expressivity > coupling'이 유의한가.
LMVD(AU, augmented) + D-Vlog(landmark, combined) 각각:
 oof 예측점수(10seed 평균) → paired bootstrap으로 ΔAUC(결합-coupling) CI + permutation.
결과: results/exp66_rigor.csv
"""
import numpy as np, warnings, csv
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
import glob
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10; NBOOT=2000; RNG=np.random.RandomState(0)
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
def oof_scores(build_feat, y):
    """build_feat(tr_idx)-> (X_all) using train-fit; returns avg oof decision scores over seeds"""
    y=np.array(y); acc=np.zeros(len(y))
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(np.zeros(len(y)),y):
            X=build_feat(tr)
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        acc+=pb
    return acc/SEEDS
def boot(y,sc_cmb,sc_cpl):
    y=np.array(y);n=len(y);d=[]
    aC=roc_auc_score(y,sc_cmb);aP=roc_auc_score(y,sc_cpl)
    for _ in range(NBOOT):
        idx=RNG.randint(0,n,n)
        if len(np.unique(y[idx]))<2:continue
        d.append(roc_auc_score(y[idx],sc_cmb[idx])-roc_auc_score(y[idx],sc_cpl[idx]))
    lo,hi=np.percentile(d,[2.5,97.5]); p=(np.sum(np.array(d)<=0)+1)/(len(d)+1)
    return aP,aC,aC-aP,lo,hi,p
out=[]
# ---- LMVD (AU): coupling vs augmented ----
print('LMVD 로딩...',flush=True)
def coupling_cov(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def aug_alpha(AUps,alpha=2.0):
    d=AUps.shape[1]; mu=AUps.mean(0); c,_=ledoit_wolf(AUps); E=c+(alpha**2)*np.outer(mu,mu)
    M=np.zeros((d+1,d+1)); M[:d,:d]=E; M[:d,d]=alpha*mu; M[d,:d]=alpha*mu; M[d,d]=1.0
    return M+1e-4*np.eye(d+1)
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423): return 1
    if (602<=i<=1116) or (1425<=i<=1824): return 0
    return None
AU_list=[];yL=[]
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
    if len(fe)>=60: AU_list.append(np.nan_to_num(np.array(fe)));yL.append(l)
yL=np.array(yL); print(f'LMVD n={len(yL)}',flush=True)
allf=np.vstack(AU_list);pm=allf.mean(0);ps=allf.std(0)+1e-6
COV=np.array([coupling_cov(a) for a in AU_list]); AUG=np.array([aug_alpha((a-pm)/ps) for a in AU_list])
def bf_cpl(tr): return TangentSpace(metric='riemann').fit(COV[tr]).transform(COV)
def bf_aug(tr): return TangentSpace(metric='riemann').fit(AUG[tr]).transform(AUG)
scC=oof_scores(bf_cpl,yL); scA=oof_scores(bf_aug,yL)
r=boot(yL,scA,scC); out.append(['LMVD',len(yL)]+list(r))
print(f'LMVD: coupling {r[0]:.3f} → augmented {r[1]:.3f}  ΔAUC {r[2]:+.3f} [95%CI {r[3]:+.3f},{r[4]:+.3f}] p={r[5]:.4f} {"★유의" if r[3]>0 else "n.s."}',flush=True)
# ---- D-Vlog (landmark): coupling vs combined ----
print('D-Vlog 로딩...',flush=True)
R=B/'D-Vlog';K=20
def nf(Vv):
    T=Vv.shape[0];P=Vv.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
Xd=[];yd=[]
for r_ in csv.DictReader(open(R/'labels.csv')):
    idx=r_['index'].strip();f=R/idx/f'{idx}_visual.npy'
    if not f.exists():continue
    try:Vv=np.load(f)
    except:continue
    if Vv.ndim!=2 or Vv.shape[0]<60 or Vv.shape[1]!=136:continue
    Xd.append(nf(Vv));yd.append(1 if r_['label'].strip().lower().startswith('depress') else 0)
yd=np.array(yd); print(f'D-Vlog n={len(yd)}',flush=True)
pcad=PCA(K).fit(np.vstack([v[::3] for v in Xd]))
def expr(Vn):
    Z=pcad.transform(Vn); spd=np.linalg.norm(np.diff(Vn,axis=0),axis=1)
    return np.array([spd.mean(),spd.std(),np.median(spd),Z.std(0).mean(),np.abs(Z-Z.mean(0)).mean()])
COVd=np.array([ (lambda c: c+1e-3*np.eye(K))(ledoit_wolf(pcad.transform(v))[0]) for v in Xd])
EXPd=np.nan_to_num(np.array([expr(v) for v in Xd]))
def bfd_cpl(tr): return TangentSpace(metric='riemann').fit(COVd[tr]).transform(COVd)
def bfd_cmb(tr):
    Xt=TangentSpace(metric='riemann').fit(COVd[tr]).transform(COVd); return np.column_stack([Xt,EXPd])
scCd=oof_scores(bfd_cpl,yd); scMd=oof_scores(bfd_cmb,yd)
r=boot(yd,scMd,scCd); out.append(['D-Vlog',len(yd)]+list(r))
print(f'D-Vlog: coupling {r[0]:.3f} → combined {r[1]:.3f}  ΔAUC {r[2]:+.3f} [95%CI {r[3]:+.3f},{r[4]:+.3f}] p={r[5]:.4f} {"★유의" if r[3]>0 else "n.s."}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp66_rigor.csv','w') as f:
    f.write('corpus,n,coupling,combined,delta,ci_lo,ci_hi,p\n')
    for r in out:f.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp66_rigor.csv',flush=True)
