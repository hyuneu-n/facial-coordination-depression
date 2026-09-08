"""영상 내부 시계열을 보존한 캐시 (버린 것 복원).
- 영상당 프레임 시퀀스 유지 (T까지, 패딩용 길이 기록)
- landmark: nf + Procrustes 정렬(위치·크기·회전 제거) + PCA20
- acoustic: 25dim 시계열 그대로 보존
- 유저당 영상 수 제한 없음 (MAXK 미적용, 전량 저장)
"""
import pickle, numpy as np, pandas as pd, os, warnings, time
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
D=20; MINF=6; TMAX=60      # 영상당 최대 프레임(=baseline과 동일 조건 확보). 원본 median 10
R=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
OUT=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache')
def nf(V):
    T=V.shape[0]; P=V.reshape(T,68,2).astype(np.float64); P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6
    return P/sc[:,None]
def align(P,ref):
    H=np.einsum('tij,ik->tjk',P,ref); U,S,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(np.einsum('tij,tjk->tik',U,Vt)))
    Dm=np.zeros_like(U); Dm[:,0,0]=1; Dm[:,1,1]=d
    return np.einsum('tij,tjk->tik',P,np.einsum('tij,tjk,tkl->til',U,Dm,Vt))
lab=pd.read_csv(f'{R}/labels.csv'); sp_of=dict(zip(lab['names'],lab['split']))
users=[]
for fn,y in [('dep_feat.pkl',1),('nondep_feat.pkl',0)]:
    with open(f'{R}/{fn}','rb') as f: d=pickle.load(f)
    for n,v in zip(d['name'],d['features']):
        if sp_of.get(n): users.append((n,y,sp_of[n],v))
    del d
rng=np.random.RandomState(0); acc=[]
for n,y,s,v in users:
    if s!='train': continue
    for i in rng.choice(len(v),size=min(len(v),3),replace=False):
        a=np.asarray(v[i])
        if a.shape[0]>=MINF: acc.append(nf(a[:,:136]).mean(0))
REF=np.mean(acc,0); REF=REF-REF.mean(0); del acc
pool=[]
for n,y,s,v in users:
    if s!='train': continue
    for i in rng.choice(len(v),size=min(len(v),25),replace=False):
        a=np.asarray(v[i])
        if a.shape[0]>=MINF: pool.append(align(nf(a[:,:136]),REF).reshape(-1,136)[::3])
pca=PCA(D).fit(np.vstack(pool)); del pool
print(f'PCA explained={pca.explained_variance_ratio_.sum():.4f}', flush=True)
# acoustic 전역 통계 (train)
astat=[]
for n,y,s,v in users:
    if s!='train': continue
    for i in rng.choice(len(v),size=min(len(v),10),replace=False):
        a=np.asarray(v[i])
        if a.shape[0]>=MINF: astat.append(a[:,136:])
AS=np.vstack(astat); amu,asd=AS.mean(0), AS.std(0)+1e-6; del astat,AS
out={}; t0=time.time(); nvid=[]
for ui,(n,y,s,v) in enumerate(users):
    vis,aud,lens=[],[],[]
    for a in v:                                  # ★ 영상 수 제한 없음
        a=np.asarray(a,dtype=np.float64)
        T=a.shape[0]
        if T<MINF: continue
        Z=pca.transform(align(nf(a[:,:136]),REF).reshape(-1,136))[:TMAX]   # [t,20] 시계열 보존
        A=((a[:,136:]-amu)/asd)[:TMAX]                                     # [t,25] 시계열 보존
        t=len(Z)
        zp=np.zeros((TMAX,D),np.float32); zp[:t]=Z
        ap=np.zeros((TMAX,25),np.float32); ap[:t]=A
        vis.append(zp); aud.append(ap); lens.append(t)
    if not vis: continue
    out[n]=dict(vis=np.asarray(vis,np.float32), aud=np.asarray(aud,np.float32),
                lens=np.asarray(lens,np.int16), label=y, split=s)
    nvid.append(len(vis))
    if (ui+1)%150==0: print(f'  {ui+1}/{len(users)} {time.time()-t0:.0f}s', flush=True)
nvid=np.array(nvid)
print(f'users={len(out)} videos total={nvid.sum()} med/user={np.median(nvid):.0f} max={nvid.max()}', flush=True)
p=f'{OUT}/mud3_seq.pkl'
with open(p,'wb') as f: pickle.dump(out,f,protocol=4)
print('saved', p, f'{os.path.getsize(p)/1e9:.2f} GB', flush=True)
