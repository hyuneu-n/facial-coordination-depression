"""
exp45 — within-session 정신운동(psychomotor) precursor 마커.
robust 신호=속도(느림). 종합 정신운동 특징 → 분류 + static coupling과 결합.
특징: 속도 mean/std/median, event rate, event 진폭, pre-event 속도, pre-event/global 비, 감속.
D-Vlog. 결과: results/exp45_psychomotor.csv
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
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); K=20; W=15; GAP=2
def norm_frames(V):
    T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
rows=list(csv.DictReader(open(R/'labels.csv')))
X=[];y=[]
for r in rows:
    idx=r['index'].strip();f=R/idx/f'{idx}_visual.npy'
    if not f.exists():continue
    try:V=np.load(f)
    except:continue
    if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136:continue
    X.append(norm_frames(V));y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
y=np.array(y); print(f'로드 {len(X)} vlog (우울{y.sum()})',flush=True)
pca=PCA(K).fit(np.vstack([v[::3] for v in X]))
FN=['spd_mean','spd_std','spd_med','event_rate','event_amp','pre_spd','pre_ratio','decel']
def feats(Vn):
    spd=np.linalg.norm(np.diff(Vn,axis=0),axis=1); g=spd.mean()+1e-6
    thr=np.percentile(spd,80); ev=[]; t=W+GAP
    while t<len(spd)-1:
        if spd[t]>=thr and spd[t]>=spd[t-1] and spd[t]>=spd[t+1]: ev.append(t);t+=W
        else:t+=1
    rate=len(ev)/len(spd); amp=np.mean([spd[e] for e in ev]) if ev else g
    pre=[spd[e-W-GAP:e-GAP].mean() for e in ev if e-W-GAP>=0]; prem=np.mean(pre) if pre else g
    decel=np.mean([spd[e]-spd[e-W-GAP:e-GAP].mean() for e in ev if e-W-GAP>=0]) if pre else 0
    f=[spd.mean(),spd.std(),np.median(spd),rate,amp,prem,prem/g,decel]
    c,_=ledoit_wolf(pca.transform(Vn)); c=c+1e-3*np.eye(K)
    return np.array(f),c
F=[];C=[]
for v in X:
    f,c=feats(v);F.append(f);C.append(c)
F=np.nan_to_num(np.array(F));C=np.array(C)
print('=== 기전(MDD vs HC) ===',flush=True)
for j,nm in enumerate(FN):
    md,hc=F[y==1,j].mean(),F[y==0,j].mean(); print(f'  {nm:11s} MDD={md:.3f} HC={hc:.3f} ({"↓" if md<hc else "↑"}우울)',flush=True)
def cv(mode):
    a=[]
    for s in range(10):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            if mode=='psm':FF=F
            else:
                t=TangentSpace(metric='riemann').fit(C[tr]);Xt=t.transform(C)
                FF=Xt if mode=='cov' else np.column_stack([Xt,F])
            sc=StandardScaler().fit(FF[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(FF[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(FF[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
print('=== AUC (10seed CV) ===',flush=True)
rp=cv('psm'); rc=cv('cov'); rb=cv('comb')
print(f'  정신운동만        {rp[0]:.3f}±{rp[1]:.3f}',flush=True)
print(f'  coupling만        {rc[0]:.3f}±{rc[1]:.3f}',flush=True)
print(f'  결합 ★            {rb[0]:.3f}±{rb[1]:.3f}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp45_psychomotor.csv','w') as f:
    f.write('feature,cv_AUC,std\n')
    for nm,(m,sd) in [('psychomotor',rp),('coupling',rc),('combined',rb)]:f.write(f'{nm},{m:.4f},{sd:.4f}\n')
print('DONE → exp45_psychomotor.csv',flush=True)
