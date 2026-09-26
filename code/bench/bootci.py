"""bench/bootci.py — 저장된 예측으로 bootstrap 신뢰구간 계산.
n이 작은 코퍼스(CMDC n=45)에서 점추정만 보고하면 근거가 약하다.
샘플을 복원추출해 F1·AUC의 95% 신뢰구간과, 두 모델 차이의 신뢰구간을 낸다.
차이의 CI가 0을 포함하지 않으면 우위가 통계적으로 뒷받침된다.
"""
import numpy as np, sys, glob, os
from sklearn.metrics import f1_score, roc_auc_score
from bench.data import LOADERS

RES = '/home/hyuneun/disk_b/🟡facial-prodrome/results/bench'


def load_pred(corpus, model, tag=''):
    fs = sorted(glob.glob(f'{RES}/pred_{corpus}_{model}{tag}_s*.npy'))
    if not fs:
        return None
    return np.mean([np.load(f) for f in fs], 0), len(fs)


def boot(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    N = len(y)
    f1s, aucs = [], []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        f1s.append(f1_score(y[idx], (p[idx] > 0.5).astype(int), zero_division=0))
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.array(f1s), np.array(aucs)


def boot_diff(y, pa, pb, n=2000, seed=0):
    """같은 부트스트랩 표본에서 두 모델 차이 -> 쌍대 비교."""
    rng = np.random.RandomState(seed)
    N = len(y)
    df, da = [], []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        yi = y[idx]
        df.append(f1_score(yi, (pa[idx] > .5).astype(int), zero_division=0)
                  - f1_score(yi, (pb[idx] > .5).astype(int), zero_division=0))
        da.append(roc_auc_score(yi, pa[idx]) - roc_auc_score(yi, pb[idx]))
    return np.array(df), np.array(da)


if __name__ == '__main__':
    corpus = sys.argv[1]
    A, B = sys.argv[2], sys.argv[3]          # A=제안(이식), B=베이스라인
    tag = sys.argv[4] if len(sys.argv) > 4 else ''
    d = LOADERS[corpus]()
    y = d['y']
    mask = np.ones(len(y), bool) if d['fold'] is None else (d['fold'] == 'test')
    ra, rb = load_pred(corpus, A, tag), load_pred(corpus, B, tag)
    if ra is None or rb is None:
        print('예측 파일 없음'); sys.exit(1)
    (pa, na), (pb, nb) = ra, rb
    y, pa, pb = y[mask], pa[mask], pb[mask]
    print(f'[{corpus}] n={len(y)} (양성 {int(y.sum())})  seed: {A}={na}, {B}={nb}')
    for name, p in [(A, pa), (B, pb)]:
        f1s, aucs = boot(y, p)
        print(f'  {name:<14} F1 {f1s.mean():.4f} [{np.percentile(f1s,2.5):.4f}, {np.percentile(f1s,97.5):.4f}]'
              f'   AUC {aucs.mean():.4f} [{np.percentile(aucs,2.5):.4f}, {np.percentile(aucs,97.5):.4f}]')
    df, da = boot_diff(y, pa, pb)
    for nm, v in [('F1', df), ('AUC', da)]:
        lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
        ok = '유의 (CI가 0 제외)' if lo > 0 or hi < 0 else '유의하지 않음 (CI가 0 포함)'
        print(f'  차이 {nm}: {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  p(>0)={np.mean(v>0):.3f}  -> {ok}')
