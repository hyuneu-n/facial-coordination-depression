"""
exp40 — D-Vlog(961, landmark 시계열, in-the-wild)로 coupling+정신운동지연 검증.
핵심 질문: 우리 coupling이 대규모·자연 데이터서 되나? 시간/정신운동이 여기선 잡히나?
visual npy = (T,136)=68 landmark x2. 프레임별 정규화(중심·스케일 제거) → 표정/형태만.
특징: (a) 정적 coupling(PCA20 공분산→Riemannian tangent) (b) 정신운동지연(landmark 속도↓)
       (c) 시변 coupling(전반vs후반 Riemannian 거리) (d) 결합.
평가: 공식 fold(train+valid→train, test→test) AUC + 10seed CV. 결과 CSV.
"""
import numpy as np, warnings, csv
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.distance import distance_riemann
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
def norm_frames(V):  # V:(T,136)->정규화 (T,136)
    T=V.shape[0]; P=V.reshape(T,68,2).astype(float)
    P=P-P.mean(1,keepdims=True)                 # 중심 제거(머리 위치)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6
    P=P/sc[:,None]                              # 스케일 제거
    return P.reshape(T,136)
rows=list(csv.DictReader(open(R/'labels.csv')))
X=[];meta=[]
for r in rows:
    idx=r['index'].strip(); f=R/idx/f'{idx}_visual.npy'
    if not f.exists(): continue
    try: V=np.load(f)
    except: continue
    if V.ndim!=2 or V.shape[0]<30 or V.shape[1]!=136: continue
    Vn=norm_frames(V)
    y=1 if r['label'].strip().lower().startswith('depress') else 0
    X.append(Vn); meta.append((idx,y,r['fold'].strip().lower()))
print(f'로드: {len(X)} vlog (우울 {sum(m[1] for m in meta)})',flush=True)
y=np.array([m[1] for m in meta]); fold=np.array([m[2] for m in meta])

# PCA(20) 전역 (빠른 신호 확인용; leak caveat)
allcat=np.vstack([v[::3] for v in X])  # 다운샘플 concat
pca=PCA(n_components=20).fit(allcat)
def feats(Vn):
    Z=pca.transform(Vn)                         # (T,20)
    c,_=ledoit_wolf(Z); c=c+1e-3*np.eye(c.shape[0])   # 정적 coupling(+reg)
    # 정신운동: landmark 속도(프레임차 크기) 평균/표준편차
    sp=np.linalg.norm(np.diff(Vn,axis=0),axis=1); psm=[sp.mean(),sp.std()]
    # 시변: 전반 vs 후반 coupling 거리
    h=len(Z)//2
    try:
        c1,_=ledoit_wolf(Z[:h]); c2,_=ledoit_wolf(Z[h:]); c1=c1+1e-3*np.eye(20); c2=c2+1e-3*np.eye(20); drift=distance_riemann(c1,c2)
    except: drift=0.0
    return c,np.array(psm+[drift])
COV=[];AUX=[]
for v in X:
    c,a=feats(v); COV.append(c); AUX.append(a)
COV=np.array(COV); AUX=np.nan_to_num(np.array(AUX))

def evalset(make_feat, name, tan=False):
    # 공식 fold
    tr=np.isin(fold,['train','valid','training']); te=fold=='test'
    if tan:
        ts=TangentSpace(metric='riemann').fit(COV[tr]); Xa=ts.transform(COV)
        F=np.column_stack([Xa, AUX]) if make_feat=='comb' else (Xa if make_feat=='cov' else AUX)
    else:
        F=AUX
    sc=StandardScaler().fit(F[tr]); clf=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(F[tr]),y[tr])
    auc_off=roc_auc_score(y[te],clf.decision_function(sc.transform(F[te]))) if te.sum()>0 and len(set(y[te]))>1 else float('nan')
    # 10seed CV(전체)
    a=[]
    for s in range(10):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tri,tei in skf.split(F,y):
            if tan and make_feat!='psm':
                ts2=TangentSpace(metric='riemann').fit(COV[tri]);Xt=ts2.transform(COV)
                FF=np.column_stack([Xt,AUX]) if make_feat=='comb' else (Xt if make_feat=='cov' else AUX)
            else: FF=F
            sc2=StandardScaler().fit(FF[tri]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc2.transform(FF[tri]),y[tri])
            pb[tei]=cl.decision_function(sc2.transform(FF[tei]))
        a.append(roc_auc_score(y,pb))
    print(f'  {name:26s} test-fold AUC={auc_off:.3f} | 10seedCV AUC={np.mean(a):.3f}±{np.std(a):.3f}',flush=True)
    return name,auc_off,np.mean(a),np.std(a)

print('=== D-Vlog 결과 (공식 fold + CV) ===',flush=True)
res=[]
res.append(evalset('cov','정적 coupling(landmark)',tan=True))
res.append(evalset('psm','정신운동지연(속도+시변)',tan=False))
res.append(evalset('comb','결합(coupling+정신운동+시변)',tan=True))
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp40_dvlog.csv','w') as f:
    f.write('feature,test_AUC,cv_AUC,cv_std\n')
    for nm,o,cv,sd in res: f.write(f'{nm},{o:.4f},{cv:.4f},{sd:.4f}\n')
print('DONE → exp40_dvlog.csv',flush=True)
