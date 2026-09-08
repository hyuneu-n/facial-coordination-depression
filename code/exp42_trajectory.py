"""
exp42 (새 primitive) — SPD manifold 위 coupling '궤적'의 기하.
sliding-window coupling → manifold 곡선 → 경로길이/속도/dispersion/가속.
가설: 우울 = 궤적이 느리고(속도↓) 좁음(dispersion↓, rigidity).
데이터: D-Vlog(landmark, 긴 시퀀스, n≈959). baseline=정적 coupling(평균 cov tangent).
결과: results/exp42_trajectory.csv
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
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
WIN,STEP,K=30,10,20
def norm_frames(V):
    T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
rows=list(csv.DictReader(open(R/'labels.csv')))
X=[];y=[];fold=[]
for r in rows:
    idx=r['index'].strip();f=R/idx/f'{idx}_visual.npy'
    if not f.exists():continue
    try:V=np.load(f)
    except:continue
    if V.ndim!=2 or V.shape[0]<WIN*3 or V.shape[1]!=136:continue
    X.append(norm_frames(V));y.append(1 if r['label'].strip().lower().startswith('depress') else 0);fold.append(r['fold'].strip().lower())
y=np.array(y);fold=np.array(fold)
print(f'로드 {len(X)} vlog (우울{y.sum()})',flush=True)
pca=PCA(K).fit(np.vstack([v[::3] for v in X]))
def covs_of(Vn):
    Z=pca.transform(Vn);cs=[]
    for a in range(0,len(Z)-WIN+1,STEP):
        c,_=ledoit_wolf(Z[a:a+WIN]);cs.append(c+1e-3*np.eye(K))
    return cs
STAT=[];TRAJ=[]
for v in X:
    cs=covs_of(v)
    M=mean_riemann(np.array(cs))
    dists=[distance_riemann(cs[i],cs[i+1]) for i in range(len(cs)-1)]
    disp=[distance_riemann(c,M) for c in cs]
    path=float(np.sum(dists)); vel=float(np.mean(dists)); dsp=float(np.mean(disp))
    acc=float(np.std(dists)); nwin=len(cs)
    STAT.append(M)                                  # 정적 baseline (평균 cov)
    TRAJ.append([path,vel,dsp,acc,np.log(nwin)])    # 궤적 기하
STAT=np.array(STAT);TRAJ=np.nan_to_num(np.array(TRAJ))

# 가설 확인: 우울 vs 정상 평균
for j,nm in enumerate(['path','velocity','dispersion','accel']):
    md,hc=TRAJ[y==1,j].mean(),TRAJ[y==0,j].mean()
    print(f'  {nm:10s} MDD={md:.3f} HC={hc:.3f}  ({"↓우울" if md<hc else "↑우울"})',flush=True)

def cv(featfn, tan=False):
    a=[]
    for s in range(10):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(TRAJ,y):
            if tan:
                ts=TangentSpace(metric='riemann').fit(STAT[tr]);F=ts.transform(STAT)
            else:F=TRAJ
            if featfn=='comb':
                ts=TangentSpace(metric='riemann').fit(STAT[tr]);F=np.column_stack([ts.transform(STAT),TRAJ])
            sc=StandardScaler().fit(F[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(F[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(F[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
print('=== AUC (10seed CV) ===',flush=True)
s_stat=cv('stat',tan=True); print(f'  정적 coupling(평균)   {s_stat[0]:.3f}±{s_stat[1]:.3f}',flush=True)
s_traj=cv('traj');          print(f'  궤적 기하(새 primitive) {s_traj[0]:.3f}±{s_traj[1]:.3f}',flush=True)
s_comb=cv('comb');          print(f'  결합                  {s_comb[0]:.3f}±{s_comb[1]:.3f}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp42_trajectory.csv','w') as f:
    f.write('feature,cv_AUC,std\n')
    for nm,(m,sd) in [('static',s_stat),('trajectory',s_traj),('combined',s_comb)]:f.write(f'{nm},{m:.4f},{sd:.4f}\n')
print('DONE → exp42_trajectory.csv',flush=True)
