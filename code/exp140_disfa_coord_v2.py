"""exp140 — DISFA 협응(coordination) 검증 v2. 정확한 인덱스 + 정규화된 상관 + 진단.
DISFA 66점 = dlib 구조와 동일(0-16 턱, 17-26 눈썹, 27-35 코, 36-47 눈) + 입만 48-65(18점, dlib은 20점).
협응강도 = 윈도우 내 landmark 좌표 표준화 후 상관계수 절대값 평균(움직임 크기와 분리된 순수 co-movement).
"""
import os, glob, re, numpy as np
from scipy.io import loadmat
from scipy import stats
D = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/DISFA')
LP = f'{D}/Landmark_Points/Landmark_Points'
AU = f'{D}/ActionUnit_Labels'
subs = sorted(os.listdir(AU))
W, STR = 40, 20   # 40프레임(~2초@20fps)

MOUTH = list(range(48,66))   # 정확한 입(18점)
JAW   = list(range(0,17))    # 턱선(대조군1)
BROW  = list(range(17,27))   # 눈썹(대조군2)

def load_au(sub, au):
    p = f'{AU}/{sub}/{sub}_au{au}.txt'
    if not os.path.exists(p): return None
    return np.array([int(l.split(',')[1]) for l in open(p) if l.strip()])

def coord_strength(P):  # P: [W, n_pts, 2] -> 정규화 상관 기반
    X = P.reshape(P.shape[0], -1).astype(np.float64)  # [W, 2*n_pts]
    sd = X.std(0)
    if (sd < 1e-6).any(): sd = np.where(sd<1e-6, 1.0, sd)
    Xz = (X - X.mean(0)) / sd
    C = np.corrcoef(Xz.T)
    C = np.nan_to_num(C)
    off = C[~np.eye(len(C), dtype=bool)]
    return np.abs(off).mean()

results = {}
diag = []
t0 = __import__('time').time()
for si, sub in enumerate(subs):
    a15, a17 = load_au(sub,15), load_au(sub,17)
    if a15 is None or a17 is None: diag.append((sub,'no_au')); continue
    lmdir = f'{LP}/{sub}/tmp_frame_lm'
    if not os.path.isdir(lmdir): diag.append((sub,'no_lmdir')); continue
    files = glob.glob(f'{lmdir}/*_lm.mat')
    idx_map = {}
    for f in files:
        m = re.search(r'_(\d+)_lm\.mat$', f)
        if m: idx_map[int(m.group(1))] = f
    if not idx_map: diag.append((sub,'no_files')); continue
    maxidx = max(idx_map.keys())
    n = min(maxidx+1, len(a15), len(a17))
    pts = np.full((n, 66, 2), np.nan)
    nload = 0
    for i in range(n):
        f = idx_map.get(i)
        if f is None: continue
        try:
            pts[i] = loadmat(f)['pts']; nload += 1
        except Exception:
            pass
    valid = ~np.isnan(pts).any((1,2))
    au_avg = (a15[:n].astype(float) + a17[:n].astype(float)) / 2

    reg_r = {}
    for name, REG in [('mouth',MOUTH), ('jaw',JAW), ('brow',BROW)]:
        cs, auw = [], []
        for s in range(0, n-W, STR):
            if valid[s:s+W].sum() < W*0.8: continue  # 80% 이상 유효
            seg = pts[s:s+W, REG]
            seg = np.nan_to_num(seg, nan=np.nanmean(seg))
            cs.append(coord_strength(seg)); auw.append(au_avg[s:s+W].mean())
        cs, auw = np.array(cs), np.array(auw)
        if len(cs) >= 10 and auw.std() > 0:
            r,_ = stats.pearsonr(cs, auw)
            reg_r[name] = (r, len(cs))
        else:
            reg_r[name] = (np.nan, len(cs))
    results[sub] = reg_r
    diag.append((sub, f'ok n={n} loaded={nload} valid={valid.sum()} windows_mouth={reg_r["mouth"][1]}'))
    if (si+1) % 5 == 0:
        print(f'  {si+1}/{len(subs)} ({__import__("time").time()-t0:.0f}s)', flush=True)

print('\n[진단: 피험자별 상태]', flush=True)
for sub, msg in diag: print(f'  {sub}: {msg}', flush=True)

print('\n[요약]', flush=True)
for name in ['mouth','jaw','brow']:
    rs = [v[name][0] for v in results.values() if not np.isnan(v[name][0])]
    print(f'  {name:6s}: n={len(rs)}명  r={np.mean(rs):.4f} ± {np.std(rs):.4f}' if rs else f'  {name}: 데이터 없음', flush=True)

mouth_rs = {k:v['mouth'][0] for k,v in results.items() if not np.isnan(v['mouth'][0])}
jaw_rs   = {k:v['jaw'][0]   for k,v in results.items() if not np.isnan(v['jaw'][0])}
common = set(mouth_rs) & set(jaw_rs)
if len(common) >= 3:
    m = np.array([mouth_rs[k] for k in common]); j = np.array([jaw_rs[k] for k in common])
    t,p = stats.ttest_rel(m, j)
    print(f'\n  mouth vs jaw paired t-test (n={len(common)}): t={t:.3f} p={p:.5f}', flush=True)
print('DONE', flush=True)
