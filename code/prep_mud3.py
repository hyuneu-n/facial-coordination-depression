"""MUD3 전처리 — 유저별 [영상 시퀀스] → 영상별 프레임협응 SPD + acoustic 요약.
우리 exp80/exp110 파이프라인(nf 정규화 → PCA → ledoit_wolf 공분산)을 영상 단위로 적용.
출력: data/MUD3/cache/mud3_d{D}.npz  (유저별 covs[K,d,d], aud[K,50], nvid, label, split, name)
"""
import pickle, numpy as np, pandas as pd, os, sys, time, warnings
from sklearn.decomposition import PCA
from sklearn.covariance import ledoit_wolf
warnings.filterwarnings('ignore')

D = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MINF = 6                      # 최소 프레임 (데이터 최소치)
R = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
OUT = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache')
os.makedirs(OUT, exist_ok=True)

def nf(V):
    """프레임별 68점 중심화 + 스케일 정규화 (평행이동·스케일 불변) — exp80 nf()와 동일"""
    T = V.shape[0]
    P = V.reshape(T, 68, 2).astype(np.float64)
    P = P - P.mean(1, keepdims=True)
    sc = np.sqrt((P**2).sum(2).mean(1, keepdims=True)) + 1e-6
    return (P / sc[:, None]).reshape(T, 136)

lab = pd.read_csv(f'{R}/labels.csv')
split_of = dict(zip(lab['names'], lab['split']))

users = []   # (name, label, split, list_of_videos)
for tag, fn, y in [('dep','dep_feat.pkl',1), ('nondep','nondep_feat.pkl',0)]:
    t0 = time.time()
    with open(f'{R}/{fn}','rb') as f: d = pickle.load(f)
    for name, vids in zip(d['name'], d['features']):
        sp = split_of.get(name)
        if sp is None: continue
        users.append((name, y, sp, vids))
    print(f'{tag} loaded {time.time()-t0:.0f}s, users so far {len(users)}', flush=True)
    del d

print(f'total users {len(users)}', flush=True)

# ---- PCA는 train 유저의 프레임에서만 fit (leakage 방지) ----
t0 = time.time()
pool = []
rng = np.random.RandomState(0)
for name, y, sp, vids in users:
    if sp != 'train': continue
    idx = rng.choice(len(vids), size=min(len(vids), 40), replace=False)
    for i in idx:
        v = np.asarray(vids[i])
        if v.shape[0] < MINF: continue
        pool.append(nf(v[:, :136])[::2])
pool = np.vstack(pool)
print(f'PCA fit pool {pool.shape} ({time.time()-t0:.0f}s)', flush=True)
pca = PCA(D).fit(pool)
print(f'PCA({D}) explained var = {pca.explained_variance_ratio_.sum():.4f}', flush=True)
del pool

# ---- 영상별 협응 공분산 + acoustic 요약 ----
t0 = time.time()
out = {}
I = np.eye(D)
n_drop = 0
for ui, (name, y, sp, vids) in enumerate(users):
    covs, auds = [], []
    for v in vids:
        v = np.asarray(v, dtype=np.float64)
        if v.shape[0] < MINF:
            n_drop += 1; continue
        Z = pca.transform(nf(v[:, :136]))
        c, _ = ledoit_wolf(Z)
        covs.append(c + 1e-3 * I)
        a = v[:, 136:]
        auds.append(np.concatenate([a.mean(0), a.std(0)]))
    if not covs: continue
    out[name] = dict(covs=np.asarray(covs, dtype=np.float32),
                     aud=np.asarray(auds, dtype=np.float32),
                     label=y, split=sp)
    if (ui+1) % 100 == 0:
        print(f'  {ui+1}/{len(users)} users, {time.time()-t0:.0f}s', flush=True)

print(f'done {time.time()-t0:.0f}s, users kept={len(out)}, videos dropped(<{MINF}f)={n_drop}', flush=True)
K = np.array([v['covs'].shape[0] for v in out.values()])
print(f'videos/user after filter: min={K.min()} med={np.median(K):.0f} max={K.max()} total={K.sum()}', flush=True)

path = f'{OUT}/mud3_d{D}.pkl'
with open(path, 'wb') as f: pickle.dump(out, f, protocol=4)
print('saved', path, f'{os.path.getsize(path)/1e6:.0f} MB', flush=True)
print('DONE', flush=True)
