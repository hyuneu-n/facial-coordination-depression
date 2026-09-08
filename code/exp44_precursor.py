"""
exp44 (goal 직결) — 세션 내 "표정 event 직전(precursor)" coupling.
아이디어: 강한 표정 움직임(event) 직전 수 초의 coupling이 우울에서 다른가 = "표정 짓기 전 패턴".
D-Vlog landmark. event=속도 피크. pre-window=event 직전 W프레임. pre-event coupling vs static vs event-coupling.
결과: results/exp44_precursor.csv
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
y=np.array(y)
print(f'로드 {len(X)} vlog (우울{y.sum()})',flush=True)
pca=PCA(K).fit(np.vstack([v[::3] for v in X]))
def reg(c): return c+1e-3*np.eye(K)
def feats(Vn):
    Z=pca.transform(Vn); spd=np.linalg.norm(np.diff(Vn,axis=0),axis=1)  # 속도
    thr=np.percentile(spd,80); events=[]
    t=W+GAP
    while t<len(spd)-1:
        if spd[t]>=thr and spd[t]>=spd[t-1] and spd[t]>=spd[t+1]:
            events.append(t); t+=W
        else: t+=1
    static,_=ledoit_wolf(Z); static=reg(static)
    pre=[Z[e-W-GAP:e-GAP] for e in events if e-W-GAP>=0]
    ev=[Z[e-W//2:e+W//2] for e in events]
    def poolcov(chunks):
        chunks=[c for c in chunks if len(c)>=5]
        if not chunks: return static
        c,_=ledoit_wolf(np.vstack(chunks)); return reg(c)
    prec=poolcov(pre); evc=poolcov(ev)
    # 정신운동: pre-window 평균 속도
    prespd=np.mean([spd[e-W-GAP:e-GAP].mean() for e in events if e-W-GAP>=0]) if events else spd.mean()
    return static,prec,evc,len(events),prespd
S=[];P=[];E=[];nev=[];psp=[]
for v in X:
    s,pr,e,n,ps=feats(v); S.append(s);P.append(pr);E.append(e);nev.append(n);psp.append(ps)
S=np.array(S);P=np.array(P);E=np.array(E);nev=np.array(nev);psp=np.array(psp)
print(f'평균 event수={nev.mean():.1f}',flush=True)
print(f'  pre-event 속도 MDD={psp[y==1].mean():.3f} HC={psp[y==0].mean():.3f} ({"↓우울" if psp[y==1].mean()<psp[y==0].mean() else "↑우울"})',flush=True)
def cvcov(C):
    a=[]
    for s in range(10):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            t=TangentSpace(metric='riemann').fit(C[tr]);Xt=t.transform(C[tr]);Xe=t.transform(C[te])
            sc=StandardScaler().fit(Xt);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(Xt),y[tr])
            pb[te]=cl.decision_function(sc.transform(Xe))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
print('=== AUC (10seed CV) ===',flush=True)
rs=cvcov(S); rp=cvcov(P); re=cvcov(E)
print(f'  static(전체)        {rs[0]:.3f}±{rs[1]:.3f}',flush=True)
print(f'  pre-event(전조) ★   {rp[0]:.3f}±{rp[1]:.3f}',flush=True)
print(f'  event-window        {re[0]:.3f}±{re[1]:.3f}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp44_precursor.csv','w') as f:
    f.write('feature,cv_AUC,std\n')
    for nm,(m,sd) in [('static',rs),('pre_event',rp),('event',re)]:f.write(f'{nm},{m:.4f},{sd:.4f}\n')
print('DONE → exp44_precursor.csv',flush=True)
