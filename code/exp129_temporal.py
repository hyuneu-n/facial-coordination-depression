"""exp129 — MUD3 시간축 탐색 (분류가 아니라 '변화' 자체를 본다).
영상이 시간순 정렬이라는 점을 이용. confound-free 표현(프레임평균 PCA20 + 오디오 유저내표준화) 사용.
Q1. 유저 내 표정 표현이 시간에 따라 체계적으로 변하는가? (초반 vs 후반)
Q2. 그 변화의 '방향/크기'가 우울군에서 다른가?
Q3. 시간 구간별 판별력 — 초반만 / 후반만 / 전체 중 어디가 유용한가? (조기탐지 근거)
"""
import os, pickle, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from scipy import stats
C=pickle.load(open(os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_minf20.pkl'),'rb'))
print(f'users={len(C)} (MINF=20 캐시)', flush=True)

def zs(a): return (a-a.mean(0))/(a.std(0)+1e-6)

rows=[]
for n,v in C.items():
    M=v['mean']; A=v['aud']            # [K,20] 표정평균, [K,50] 오디오
    K=len(M)
    if K<10: continue
    Mz=zs(M); Az=zs(A)                 # 유저 내 표준화 = confound 제거
    h=K//2
    early, late = Mz[:h], Mz[h:]
    # 변화 지표
    drift = np.linalg.norm(late.mean(0)-early.mean(0))            # 초반→후반 이동 크기
    var_e, var_l = early.var(0).mean(), late.var(0).mean()        # 표현 변동성
    # 시간 추세 (프레임평균 각 축의 시간 회귀 기울기 크기)
    t=np.arange(K); tz=(t-t.mean())/(t.std()+1e-9)
    slope=np.array([np.polyfit(tz, Mz[:,c],1)[0] for c in range(Mz.shape[1])])
    # 연속 영상 간 변화량 (표현 불안정성)
    step=np.linalg.norm(np.diff(Mz,axis=0),axis=1).mean()
    rows.append(dict(name=n, y=v['label'], split=v['split'], K=K,
                     drift=drift, var_e=var_e, var_l=var_l, dvar=var_l-var_e,
                     slope_mag=np.linalg.norm(slope), step=step))
import pandas as pd
df=pd.DataFrame(rows); print(f'분석 대상 {len(df)}명', flush=True)

print('\n[Q1/Q2] 시간 변화 지표의 군간 차이 (train+val, t-test)', flush=True)
d=df[df.split!='test']
print(f"{'지표':12s} {'비우울':>10s} {'우울':>10s} {'t':>8s} {'p':>9s}", flush=True)
for k in ['drift','var_e','var_l','dvar','slope_mag','step','K']:
    a=d[d.y==0][k]; b=d[d.y==1][k]
    t,p=stats.ttest_ind(a,b,equal_var=False)
    star='  <<<' if p<0.05 else ''
    print(f'{k:12s} {a.mean():10.4f} {b.mean():10.4f} {t:8.3f} {p:9.5f}{star}', flush=True)

print('\n[Q3] 시간 구간별 판별력 (표정평균+오디오, 유저내표준화, 로지스틱)', flush=True)
def seg_feat(frac_lo, frac_hi):
    X,Y,S=[],[],[]
    for n,v in C.items():
        M=v['mean']; A=v['aud']; K=len(M)
        if K<10: continue
        Mz=zs(M); Az=zs(A)
        lo,hi=int(K*frac_lo), max(int(K*frac_hi),int(K*frac_lo)+1)
        m,a=Mz[lo:hi],Az[lo:hi]
        X.append(np.concatenate([m.mean(0),m.std(0),a.mean(0),a.std(0)]))
        Y.append(v['label']); S.append(v['split'])
    X=np.array(X); Y=np.array(Y); S=np.array(S)
    return X,Y,S
print(f"{'구간':16s} {'test F1':>9s} {'AUC':>8s}", flush=True)
for tag,(lo,hi) in [('앞 25%',(0,.25)),('앞 50%',(0,.5)),('뒤 50%',(.5,1.)),('뒤 25%',(.75,1.)),('전체',(0,1.))]:
    X,Y,S=seg_feat(lo,hi)
    tr,te=S!='test',S=='test'
    sc=StandardScaler().fit(X[tr])
    m=LogisticRegression(max_iter=4000,class_weight='balanced',C=0.1).fit(sc.transform(X[tr]),Y[tr])
    p=m.predict(sc.transform(X[te])); pr=m.predict_proba(sc.transform(X[te]))[:,1]
    print(f'{tag:16s} {f1_score(Y[te],p):9.4f} {roc_auc_score(Y[te],pr):8.4f}', flush=True)
print('DONE', flush=True)
