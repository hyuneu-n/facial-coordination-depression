"""bench/data.py — 코퍼스 통일 로더.
반환 규약: dict(Xv=(N,T,Dv) f32, Xa=(N,T,Da) f32, y=(N,) int, fold=(N,) str or None)
fold=None 이면 run.py 가 stratified CV 로 분할한다.
"""
import numpy as np, csv
from pathlib import Path

ROOT = Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
T = 256


def resamp(X, t=T):
    n = len(X)
    if n == t:
        return X
    idx = np.linspace(0, n - 1, t)
    ar = np.arange(n)
    return np.stack([np.interp(idx, ar, X[:, c]) for c in range(X.shape[1])], 1)


def zs(X):
    sd = X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - X.mean(0)) / sd


def norm_face(V):
    """68 landmark: 프레임별 중심정렬 + 스케일정규화 (exp97 nf 와 동일)."""
    T2 = V.shape[0]
    P = V.reshape(T2, 68, 2).astype(float)
    P = P - P.mean(1, keepdims=True)
    sc = np.sqrt((P ** 2).sum(2).mean(1, keepdims=True)) + 1e-6
    return (P / sc[:, None]).reshape(T2, 136)


# ---------------------------------------------------------------- D-Vlog
def load_dvlog(norm=True):
    R = ROOT / 'D-Vlog'
    Xv, Xvr, Xa, y, fold = [], [], [], [], []
    Vvar, Avar = [], []
    for r in csv.DictReader(open(R / 'labels.csv')):
        idx = r['index'].strip()
        f = r.get('fold', '').strip().lower()
        if 'val' in f:
            f = 'valid'
        if f not in ('train', 'valid', 'test'):
            continue
        fv, fa = R / idx / f'{idx}_visual.npy', R / idx / f'{idx}_acoustic.npy'
        if not (fv.exists() and fa.exists()):
            continue
        try:
            V = np.load(fv)
            A = np.nan_to_num(np.load(fa).astype(float))
        except Exception:
            continue
        if V.ndim != 2 or V.shape[0] < 60 or V.shape[1] != 136:
            continue
        if A.ndim != 2 or len(A) < 10:
            continue
        v = norm_face(V) if norm else np.nan_to_num(V.astype(float))
        Xv.append(resamp(zs(v)))
        Xvr.append(resamp(v))          # z-score 없이 (exp97 충실도)
        Xa.append(resamp(zs(A)))
        Vvar.append(v.astype(np.float32))    # 원본 길이 (조기탐지 절단용)
        Avar.append(A.astype(np.float32))
        y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
        fold.append(f)
    return dict(Xv=np.asarray(Xv, np.float32), Xa=np.asarray(Xa, np.float32),
                Xv_raw=np.asarray(Xvr, np.float32),
                Xv_var=Vvar, Xa_var=Avar,
                y=np.asarray(y), fold=np.asarray(fold))


LOADERS = {'dvlog': load_dvlog}


def _cached(name, fn):
    """npz 캐시가 있으면 그걸 쓴다 (LMVD 원본 로딩 207s -> 수초)."""
    def g(*a, **k):
        import numpy as _np
        c = ROOT / f'cache_{name}.npz'
        if c.exists() and not a and not k:
            z = _np.load(c, allow_pickle=True)
            return dict(Xv=z['Xv'], Xa=z['Xa'], y=z['y'],
                        Xv_raw=z['Xv_raw'] if 'Xv_raw' in z.files else None,
                        fold=z['fold'] if 'fold' in z.files else None)
        return fn(*a, **k)
    return g


def _register_extra():
    from bench import data_extra as E
    LOADERS['lmvd'] = _cached('lmvd', E.load_lmvd)
    LOADERS['cmdc'] = _cached('cmdc', E.load_cmdc)
    LOADERS['edaic'] = _cached('edaic', E.load_edaic)
    LOADERS['lmvd_lmk'] = _cached('lmvd_lmk', E.load_lmvd_lmk)
    LOADERS['cmdc_lmk'] = _cached('cmdc_lmk', E.load_cmdc_lmk)


_register_extra()


def _register_mud3():
    from bench import data_mud3 as M
    LOADERS['mud3'] = _cached('mud3', lambda: M.load_mud3(False))
    LOADERS['mud3_aligned'] = _cached('mud3_aligned', lambda: M.load_mud3(True))
    LOADERS['mud3_rawcoord'] = _cached('mud3_rawcoord', lambda: M.load_mud3(False, normalize=False))


_register_mud3()
