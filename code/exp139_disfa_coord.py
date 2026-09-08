"""exp139 — 협응(coordination) 방법 자체를 DISFA 실측 AU 강도로 검증.
가설: 입/턱 영역 landmark의 '협응 강도'(공분산 비대각 절대값 평균)가 AU15+17 실제 강도와 상관 → 우리 방법이 진짜 AU활성을 포착.
대조: 다른 얼굴 영역(눈썹/코 부근)의 협응 강도는 AU15+17과 상관이 약해야 함(특이성).
66점 landmark: 마지막 20점=입 주변(휴리스틱, DISFA AAM 관례), 0~16=턱선, 17~30=눈썹/코 대조군.
"""
import os, glob, numpy as np
from scipy.io import loadmat
from scipy import stats
D = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/DISFA')
LP = f'{D}/Landmark_Points/Landmark_Points'
AU = f'{D}/ActionUnit_Labels'
subs = sorted(os.listdir(AU))
W, STR = 20, 10   # 20프레임(~1초@20fps) 윈도우

MOUTH = list(range(46,66))   # 마지막 20점
CTRL  = list(range(17,31))   # 눈썹/코 부근(대조군)

def load_au(sub, au):
    p = f'{AU}/{sub}/{sub}_au{au}.txt'
    if not os.path.exists(p): return None
    return np.array([int(l.split(',')[1]) for l in open(p) if l.strip()])

def coord_strength(P):  # P: [W, n_pts, 2]
    X = P.reshape(P.shape[0], -1)  # [W, 2*n_pts]
    X = X - X.mean(0)
    C = np.cov(X.T)
    off = C[~np.eye(len(C), dtype=bool)]
    return np.abs(off).mean()

mouth_r, ctrl_r = [], []
t0 = __import__('time').time()
for si, sub in enumerate(subs):
    a15, a17 = load_au(sub,15), load_au(sub,17)
    if a15 is None or a17 is None: continue
    lmdir = f'{LP}/{sub}/tmp_frame_lm'
    if not os.path.isdir(lmdir): continue
    files = sorted(glob.glob(f'{lmdir}/*_lm.mat'))
    n = min(len(files), len(a15), len(a17))
    if n < 100: continue
    pts = np.zeros((n, 66, 2))
    for i in range(n):
        try:
            pts[i] = loadmat(f'{lmdir}/{sub}_{i:04d}_lm.mat')['pts']
        except Exception:
            pts[i] = np.nan
    valid = ~np.isnan(pts).any((1,2))
    au_avg = (a15[:n].astype(float) + a17[:n].astype(float)) / 2

    mc, cc, auw = [], [], []
    for s in range(0, n-W, STR):
        if not valid[s:s+W].all(): continue
        mc.append(coord_strength(pts[s:s+W, MOUTH]))
        cc.append(coord_strength(pts[s:s+W, CTRL]))
        auw.append(au_avg[s:s+W].mean())
    if len(mc) < 10: continue
    mc, cc, auw = np.array(mc), np.array(cc), np.array(auw)
    if auw.std() > 0:
        rm,_ = stats.pearsonr(mc, auw); rc,_ = stats.pearsonr(cc, auw)
        mouth_r.append(rm); ctrl_r.append(rc)
    if (si+1) % 5 == 0:
        print(f'  {si+1}/{len(subs)} 처리 ({__import__("time").time()-t0:.0f}s)', flush=True)

mouth_r, ctrl_r = np.array(mouth_r), np.array(ctrl_r)
print(f'\n분석대상 {len(mouth_r)}명', flush=True)
print(f'입 영역 협응강도 vs AU15+17: r={mouth_r.mean():.4f} ± {mouth_r.std():.4f}', flush=True)
print(f'대조 영역 협응강도 vs AU15+17: r={ctrl_r.mean():.4f} ± {ctrl_r.std():.4f}', flush=True)
t,p = stats.ttest_rel(mouth_r, ctrl_r)
print(f'paired t-test (입 > 대조): t={t:.3f} p={p:.5f}', flush=True)
print('DONE', flush=True)
