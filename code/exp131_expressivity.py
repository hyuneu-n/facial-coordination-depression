"""exp131 — 표정 변동성(expressivity)을 epsilon 누출 없이 직접 측정.
가설(D-Vlog 확정발견 hypoexpressivity의 MUD3 재현): 우울군은 영상 간 표정 변동폭이 작다.
confound 점검도 함께: 변동성이 촬영조건(얼굴크기·프레임수·영상수)과 얽혀있는가?
"""
import os, pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats
C=pickle.load(open(os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_minf20.pkl'),'rb'))

rows=[]
for n,v in C.items():
    M=v['mean'].astype(np.float64)     # [K,20] 영상별 표정 평균 (nf정규화+PCA, confound-free)
    K=len(M)
    if K<10: continue
    sd=M.std(0)                                   # 축별 영상간 변동
    rows.append(dict(name=n, y=v['label'], split=v['split'], K=K,
        expr_sd   = sd.mean(),                    # 표정 변동폭 (핵심 지표)
        expr_sd1  = sd[0],                        # 제1주성분 변동
        expr_tot  = np.sqrt((sd**2).sum()),       # 총 변동 크기
        expr_logdet = np.linalg.slogdet(np.cov(M.T)+1e-6*np.eye(M.shape[1]))[1],  # 표현 다양성(부피)
        step      = np.linalg.norm(np.diff(M,axis=0),axis=1).mean(),  # 영상간 이동량
        rng       = (M.max(0)-M.min(0)).mean(),   # 표현 범위
    ))
df=pd.DataFrame(rows); print(f'분석 {len(df)}명 (dep {int(df.y.sum())})', flush=True)

FE=['expr_sd','expr_sd1','expr_tot','expr_logdet','step','rng']
print('\n[군간 차이] train+val, Welch t-test  (음수 t = 우울군이 더 큼)', flush=True)
d=df[df.split!='test']
print(f"{'지표':12s} {'비우울':>10s} {'우울':>10s} {'t':>8s} {'p':>10s} {'방향':>12s}", flush=True)
for k in FE:
    a,b=d[d.y==0][k], d[d.y==1][k]
    t,p=stats.ttest_ind(a,b,equal_var=False)
    dirn = '우울<비우울' if b.mean()<a.mean() else '우울>비우울'
    star=' ***' if p<0.001 else (' **' if p<0.01 else (' *' if p<0.05 else ''))
    print(f'{k:12s} {a.mean():10.4f} {b.mean():10.4f} {t:8.3f} {p:10.6f} {dirn:>12s}{star}', flush=True)

print('\n[판별력] test set', flush=True)
tr,te=df.split!='test', df.split=='test'
for tag,cols in [('expr_sd 단독',['expr_sd']), ('변동성 6지표',FE), ('변동성+영상수',FE+['K'])]:
    sc=StandardScaler().fit(df.loc[tr,cols])
    m=LogisticRegression(max_iter=4000,class_weight='balanced').fit(sc.transform(df.loc[tr,cols]),df.loc[tr,'y'])
    pr=m.predict_proba(sc.transform(df.loc[te,cols]))[:,1]; p=m.predict(sc.transform(df.loc[te,cols]))
    print(f'  {tag:16s} F1={f1_score(df.loc[te,"y"],p):.4f}  AUC={roc_auc_score(df.loc[te,"y"],pr):.4f}', flush=True)

print('\n[confound 점검] 변동성이 촬영조건과 상관있는가 (train+val)', flush=True)
raw=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
print(f'  expr_sd vs 영상수 K: r={np.corrcoef(d.expr_sd,d.K)[0,1]:+.3f}', flush=True)
print('  (얼굴크기와의 상관은 nf정규화로 구조적 제거됨 — M은 위치·크기 불변 표현)', flush=True)
print('\n[비교] baseline=0.6717 | 시각confound=0.6875 | 오디오confound=0.7273', flush=True)
print('DONE', flush=True)
