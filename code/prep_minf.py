"""프레임 수 하한(MINF)별 캐시 생성: 협응 추정에 프레임이 부족한 게 원인인지 검증용.
covs(협응) + mean(평균) 둘 다 같은 영상 집합에서 생성 → 공정 비교."""
import pickle, numpy as np, pandas as pd, os, sys, warnings
from sklearn.decomposition import PCA
from sklearn.covariance import ledoit_wolf
warnings.filterwarnings('ignore')
MINF=int(sys.argv[1]); D=20
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
        if sp_of.get(n): users.append((n,y,sp_of[n],v))
    del d
rng=np.random.RandomState(0); pool=[]
for n,y,s,v in users:
    if s!='train': continue
    for i in rng.choice(len(v), size=min(len(v),40), replace=False):
        a=np.asarray(v[i])
        if a.shape[0]>=MINF: pool.append(nf(a[:,:136])[::2])
pca=PCA(D).fit(np.vstack(pool)); del pool
out={}; I=np.eye(D); nk=[]
for n,y,s,v in users:
    covs,means,auds=[],[],[]
    for a in v:
        a=np.asarray(a,dtype=np.float64)
        if a.shape[0]<MINF: continue
        Z=pca.transform(nf(a[:,:136]))
        c,_=ledoit_wolf(Z); covs.append(c+1e-3*I); means.append(Z.mean(0))
        ac=a[:,136:]; auds.append(np.concatenate([ac.mean(0),ac.std(0)]))
    if not covs: continue
    out[n]=dict(covs=np.asarray(covs,np.float32), mean=np.asarray(means,np.float32),
                aud=np.asarray(auds,np.float32), label=y, split=s)
    nk.append(len(covs))
nk=np.array(nk)
print(f'MINF={MINF}: users={len(out)}/{len(users)}  videos/user med={np.median(nk):.0f} total={nk.sum()}', flush=True)
print(f'  split: ' + str({k: sum(1 for v in out.values() if v['split']==k) for k in ['train','val','test']}), flush=True)
print(f'  test 양성={sum(1 for v in out.values() if v["split"]=="test" and v["label"]==1)}', flush=True)
with open(f'{OUT}/mud3_minf{MINF}.pkl','wb') as f: pickle.dump(out,f,protocol=4)
print('saved', flush=True)
