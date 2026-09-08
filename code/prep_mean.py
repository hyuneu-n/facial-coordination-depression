"""no_coord ablation용: 영상별 프레임 평균(협응 없음) 캐시"""
import pickle, numpy as np, pandas as pd, os, warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
D=20; MINF=6
R=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
OUT=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache')
def nf(V):
    T=V.shape[0]; P=V.reshape(T,68,2).astype(np.float64); P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6
    return (P/sc[:,None]).reshape(T,136)
lab=pd.read_csv(f'{R}/labels.csv'); sp_of=dict(zip(lab['names'],lab['split']))
users=[]
for fn,y in [('dep_feat.pkl',1),('nondep_feat.pkl',0)]:
    with open(f'{R}/{fn}','rb') as f: d=pickle.load(f)
    for n,v in zip(d['name'],d['features']):
        if sp_of.get(n) is not None: users.append((n,sp_of[n],v))
    del d
rng=np.random.RandomState(0); pool=[]
for n,s,v in users:
    if s!='train': continue
    for i in rng.choice(len(v), size=min(len(v),40), replace=False):
        a=np.asarray(v[i])
        if a.shape[0]>=MINF: pool.append(nf(a[:,:136])[::2])
pca=PCA(D).fit(np.vstack(pool)); del pool
out={}
for n,s,v in users:
    z=[]
    for a in v:
        a=np.asarray(a,dtype=np.float64)
        if a.shape[0]<MINF: continue
        z.append(pca.transform(nf(a[:,:136])).mean(0))
    if z: out[n]=np.asarray(z,dtype=np.float32)
with open(f'{OUT}/mud3_mean20.pkl','wb') as f: pickle.dump(out,f,protocol=4)
print('saved', len(out), flush=True)
