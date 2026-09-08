"""
exp65 — D-Vlog(2번째 대형 in-the-wild, landmark)서 'expressivity가 coupling에 더해지나' 재현.
LMVD(AU)의 tension = D-Vlog(landmark)의 표정 움직임 활성도(expressivity)로 대응.
 둘 다 "얼굴이 얼마나 표현적인가"(flat affect). coupling은 이 활성도를 정규화로 버림.
질문: expressivity(움직임 활성도)가 coupling에 더해지나? 기전(MDD<HC)? + permutation.
결과: results/exp65_dvlog.csv
"""
import numpy as np, warnings, csv
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); K=20; SEEDS=10
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
def expressivity(Vn):  # 표정 움직임 활성도(=tension 대응)
    Z=pca.transform(Vn); spd=np.linalg.norm(np.diff(Vn,axis=0),axis=1)
    return np.array([spd.mean(),spd.std(),np.median(spd), Z.std(0).mean(), np.abs(Z-Z.mean(0)).mean()])
def coupling(Vn):
    c,_=ledoit_wolf(pca.transform(Vn)); return c+1e-3*np.eye(K)
COV=[coupling(v) for v in X]; EXP=np.array([expressivity(v) for v in X])
# 기전
names=['spd_mean','spd_std','spd_med','pc_std','pc_dev']
print('=== 기전(expressivity MDD vs HC) ===',flush=True)
for j,nm in enumerate(names):
    md,hc=EXP[y==1,j].mean(),EXP[y==0,j].mean();_,p=mannwhitneyu(EXP[y==1,j],EXP[y==0,j])
    print(f'  {nm:9s} MDD={md:.3f} HC={hc:.3f} ({"↓우울(flat)" if md<hc else "↑"}) p={p:.4f}',flush=True)
def cv(mode):
    COVa=np.array(COV);a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(COVa,y):
            if mode=='exp': Xf=EXP
            else:
                T=TangentSpace(metric='riemann').fit(COVa[tr]);Xt=T.transform(COVa)
                Xf=Xt if mode=='cov' else np.column_stack([Xt,EXP])
            sc=StandardScaler().fit(Xf[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(Xf[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(Xf[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
re=cv('exp');rc=cv('cov');rb=cv('comb')
print('=== AUC (10seed) ===',flush=True)
print(f'  expressivity 단독 {re[0]:.3f}±{re[1]:.3f}',flush=True)
print(f'  coupling         {rc[0]:.3f}±{rc[1]:.3f}',flush=True)
print(f'  결합             {rb[0]:.3f}±{rb[1]:.3f}  Δ{rb[0]-rc[0]:+.3f} {"★expressivity 추가" if rb[0]>rc[0]+0.015 else "추가 미미"}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp65_dvlog.csv','w') as f:
    f.write('feature,AUC,std\n')
    for nm,r in [('expressivity',re),('coupling',rc),('combined',rb)]:f.write(f'{nm},{r[0]:.4f},{r[1]:.4f}\n')
print('DONE → exp65_dvlog.csv',flush=True)
