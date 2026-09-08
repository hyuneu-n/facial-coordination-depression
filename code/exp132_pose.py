"""exp132 — expr_sd가 '표정 변동'인가 '촬영 각도(pose) 변동'인가?
Procrustes 정렬(회전 제거) + pose 대리지표 분리로 검증.
"""
import os, pickle, numpy as np, pandas as pd, warnings
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats
warnings.filterwarnings('ignore')
R=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
MINF=20

def nf(V):   # 위치·크기 제거 (기존)
    T=V.shape[0]; P=V.reshape(T,68,2).astype(np.float64); P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6
    return P/sc[:,None]                     # [T,68,2]

def procrustes_align(P, ref):
    """회전까지 제거: 각 프레임을 ref에 최적 회전 정렬 (Kabsch)"""
    H = np.einsum('tij,ik->tjk', P, ref)    # [T,2,2]
    U,S,Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum('tij,tjk->tik', U, Vt)))
    D = np.zeros_like(U); D[:,0,0]=1; D[:,1,1]=d
    Rm = np.einsum('tij,tjk,tkl->til', U, D, Vt)
    return np.einsum('tij,tjk->tik', P, Rm)

lab=pd.read_csv(f'{R}/labels.csv'); sp_of=dict(zip(lab['names'],lab['split']))
users=[]
for fn,y in [('dep_feat.pkl',1),('nondep_feat.pkl',0)]:
    with open(f'{R}/{fn}','rb') as f: d=pickle.load(f)
    for n,v in zip(d['name'],d['features']):
        if sp_of.get(n): users.append((n,y,sp_of[n],v))
    del d
print(f'users={len(users)}', flush=True)

# 공통 참조 얼굴 (train 평균)
acc=[]
rng_=np.random.RandomState(0)
for n,y,s,v in users[:200]:
    for i in rng_.choice(len(v), size=min(len(v),5), replace=False):
        a=np.asarray(v[i])
        if a.shape[0]>=MINF: acc.append(nf(a[:,:136]).mean(0))
REF=np.mean(acc,0); REF=REF-REF.mean(0)

rows=[]
for n,y,s,v in users:
    raw_means, ali_means, poses = [], [], []
    for a in v:
        a=np.asarray(a,dtype=np.float64)
        if a.shape[0]<MINF: continue
        P=nf(a[:,:136])                       # [T,68,2] 위치·크기 제거
        raw_means.append(P.mean(0).ravel())   # 정렬 전 영상 평균 형태
        Pa=procrustes_align(P, REF)           # 회전까지 제거
        ali_means.append(Pa.mean(0).ravel())
        # pose 대리지표: 정렬에 필요한 회전각 + 좌우 비대칭
        m=P.mean(0)
        ang=np.arctan2(m[36:42,1].mean()-m[42:48,1].mean(), m[36:42,0].mean()-m[42:48,0].mean())
        asym=abs(np.linalg.norm(m[36:42].mean(0)-m[30])-np.linalg.norm(m[42:48].mean(0)-m[30]))
        poses.append([ang, asym])
    if len(raw_means)<10: continue
    Rm=np.array(raw_means); Am=np.array(ali_means); Po=np.array(poses)
    rows.append(dict(name=n,y=y,split=s,K=len(Rm),
        sd_raw = Rm.std(0).mean(),          # 정렬 전 변동 (기존 expr_sd에 해당)
        sd_ali = Am.std(0).mean(),          # 회전 제거 후 변동 (순수 표정에 가까움)
        sd_pose_ang = Po[:,0].std(),        # 촬영 각도 변동
        sd_pose_asym= Po[:,1].std(),        # 비대칭(고개돌림) 변동
    ))
df=pd.DataFrame(rows); print(f'분석 {len(df)}명', flush=True)

print('\n[군간 차이] train+val', flush=True)
d=df[df.split!='test']
print(f"{'지표':14s} {'비우울':>10s} {'우울':>10s} {'t':>8s} {'p':>10s}", flush=True)
for k in ['sd_raw','sd_ali','sd_pose_ang','sd_pose_asym']:
    a,b=d[d.y==0][k], d[d.y==1][k]
    t,p=stats.ttest_ind(a,b,equal_var=False)
    star=' ***' if p<0.001 else (' **' if p<0.01 else (' *' if p<0.05 else ''))
    print(f'{k:14s} {a.mean():10.5f} {b.mean():10.5f} {t:8.3f} {p:10.6f}{star}', flush=True)

print('\n[판별력] test', flush=True)
tr,te=df.split!='test', df.split=='test'
for tag,cols in [('sd_raw(정렬전)',['sd_raw']), ('sd_ali(회전제거)',['sd_ali']),
                 ('pose만',['sd_pose_ang','sd_pose_asym']),
                 ('sd_ali + pose',['sd_ali','sd_pose_ang','sd_pose_asym'])]:
    sc=StandardScaler().fit(df.loc[tr,cols])
    m=LogisticRegression(max_iter=4000,class_weight='balanced').fit(sc.transform(df.loc[tr,cols]),df.loc[tr,'y'])
    pr=m.predict_proba(sc.transform(df.loc[te,cols]))[:,1]; p=m.predict(sc.transform(df.loc[te,cols]))
    print(f'  {tag:18s} F1={f1_score(df.loc[te,"y"],p):.4f}  AUC={roc_auc_score(df.loc[te,"y"],pr):.4f}', flush=True)

print(f'\n[상관] sd_raw vs pose_ang r={np.corrcoef(d.sd_raw,d.sd_pose_ang)[0,1]:+.3f} | sd_ali vs pose_ang r={np.corrcoef(d.sd_ali,d.sd_pose_ang)[0,1]:+.3f}', flush=True)
print('DONE', flush=True)
