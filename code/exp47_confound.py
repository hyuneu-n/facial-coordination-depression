"""
exp47 — D-Vlog 정신운동 신호가 녹화 scale confound인지 검증.
raw(크기 유지) vs per-subject z-score(크기 제거) 속도특징 AUC 비교.
z-score하면 AUC가 chance로 떨어지면 → 신호는 '크기(confound)'였음.
결과: results/exp47_confound.csv
"""
import numpy as np, warnings, csv
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); W=15; GAP=2
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
def psm(Vn, zscore):
    Z=(Vn-Vn.mean(0))/(Vn.std(0)+1e-6) if zscore else Vn
    spd=np.linalg.norm(np.diff(Z,axis=0),axis=1); g=spd.mean()+1e-6
    thr=np.percentile(spd,80); ev=[];t=W+GAP
    while t<len(spd)-1:
        if spd[t]>=thr and spd[t]>=spd[t-1] and spd[t]>=spd[t+1]:ev.append(t);t+=W
        else:t+=1
    rate=len(ev)/len(spd); amp=np.mean([spd[e] for e in ev]) if ev else g
    pre=[spd[e-W-GAP:e-GAP].mean() for e in ev if e-W-GAP>=0]; prem=np.mean(pre) if pre else g
    decel=np.mean([spd[e]-spd[e-W-GAP:e-GAP].mean() for e in ev if e-W-GAP>=0]) if pre else 0
    return np.array([spd.mean(),spd.std(),np.median(spd),rate,amp,prem,prem/g,decel])
def auc(F):
    F=np.nan_to_num(F);a=[]
    for s in range(10):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            sc=StandardScaler().fit(F[tr]);cl=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(F[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(F[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
Fraw=np.array([psm(v,False) for v in X]); Fz=np.array([psm(v,True) for v in X])
r1=auc(Fraw); r2=auc(Fz)
# spd_mean 단독도
s1=auc(Fraw[:,[0]]); s2=auc(Fz[:,[0]])
print(f'  raw(크기유지)    전체 AUC={r1[0]:.3f} | spd_mean단독 {s1[0]:.3f}',flush=True)
print(f'  z-score(크기제거) 전체 AUC={r2[0]:.3f} | spd_mean단독 {s2[0]:.3f}',flush=True)
print(f'  → {"CONFOUND 확인(z-score시 붕괴)" if r2[0]<r1[0]-0.05 else "confound 아님(z-score도 유지)"}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp47_confound.csv','w') as f:
    f.write('setting,all_AUC,spdmean_AUC\n')
    f.write(f'raw,{r1[0]:.4f},{s1[0]:.4f}\nzscore,{r2[0]:.4f},{s2[0]:.4f}\n')
print('DONE → exp47_confound.csv',flush=True)
