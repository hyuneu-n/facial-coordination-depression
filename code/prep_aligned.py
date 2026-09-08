"""Procrustes 정렬(위치·크기·회전 모두 제거) 캐시 — 진짜 confound-free 표정 표현."""
import pickle, numpy as np, pandas as pd, os, warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
D=20; MINF=6
R=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
OUT=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache')
def nf(V):
    T=V.shape[0]; P=V.reshape(T,68,2).astype(np.float64); P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6
    return P/sc[:,None]
def align(P, ref):
    H=np.einsum('tij,ik->tjk',P,ref); U,S,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(np.einsum('tij,tjk->tik',U,Vt)))
    Dm=np.zeros_like(U); Dm[:,0,0]=1; Dm[:,1,1]=d
    Rm=np.einsum('tij,tjk,tkl->til',U,Dm,Vt)
    return np.einsum('tij,tjk->tik',P,Rm)
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
    for i in rng.choice(len(v), size=min(len(v),3), replace=False):
        a=np.asarray(v[i])
        if a.shape[0]>=MINF: acc.append(nf(a[:,:136]).mean(0))
REF=np.mean(acc,0); REF=REF-REF.mean(0); del acc
print('REF built', flush=True)
pool=[]
for n,y,s,v in users:
    if s!='train': continue
    for i in rng.choice(len(v), size=min(len(v),30), replace=False):
        a=np.asarray(v[i])
        if a.shape[0]>=MINF: pool.append(align(nf(a[:,:136]),REF).reshape(-1,136)[::3])
pca=PCA(D).fit(np.vstack(pool))
print(f'PCA fit, explained={pca.explained_variance_ratio_.sum():.4f}', flush=True)
del pool
out={}
for ui,(n,y,s,v) in enumerate(users):
    means,auds=[],[]
    for a in v:
        a=np.asarray(a,dtype=np.float64)
        if a.shape[0]<MINF: continue
        Z=pca.transform(align(nf(a[:,:136]),REF).reshape(-1,136))
        means.append(Z.mean(0))
        ac=a[:,136:]; auds.append(np.concatenate([ac.mean(0),ac.std(0)]))
    if not means: continue
    out[n]=dict(mean=np.asarray(means,np.float32), aud=np.asarray(auds,np.float32), label=y, split=s)
    if (ui+1)%150==0: print(f'  {ui+1}/{len(users)}', flush=True)
with open(f'{OUT}/mud3_aligned.pkl','wb') as f: pickle.dump(out,f,protocol=4)
print('saved', len(out), flush=True)
