"""bench/data_extra.py — LMVD / CMDC / E-DAIC 로더. data.py 가 import 해 LOADERS 에 등록한다.

공통 설계
- visual: OpenFace per-frame CSV -> timestamp 기준 **초당 평균**으로 축약 -> T=256 리샘플
  (프레임 수가 코퍼스마다 수천~수만이라 초당 축약이 없으면 메모리·시간이 감당 안 된다.
   exp107 의 persec() 와 동일한 축약이므로 기존 결과와 연속성이 유지된다.)
- acoustic: 코퍼스가 주는 형태를 그대로 쓰고 T=256 리샘플.
"""
import numpy as np, glob, os, re, csv
import pandas as pd
from pathlib import Path

from bench.data import ROOT, T, resamp, zs, norm_face

# OpenFace AU intensity 17종 (exp107 과 동일 목록)
AUS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
       'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
       'AU25_r', 'AU26_r', 'AU45_r']
GAZE = ['gaze_angle_x', 'gaze_angle_y']
POSE = ['pose_Rx', 'pose_Ry', 'pose_Rz']
# OpenFace 2D landmark 68점 (x_0..x_67, y_0..y_67) — D-Vlog/MUD3 와 동일 규격
LMK = [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)]


def _lmk_order(have):
    """_persec 이 돌려준 컬럼 순서를 (x0,y0,x1,y1,...) 이 아닌 (x0..x67,y0..y67) 로 맞춘다.
    norm_face 는 reshape(T,68,2) 를 하므로 x,y 가 교대로 와야 한다."""
    idx = {c: i for i, c in enumerate(have)}
    order = []
    for i in range(68):
        if f'x_{i}' in idx and f'y_{i}' in idx:
            order += [idx[f'x_{i}'], idx[f'y_{i}']]
    return order


def _persec(path, cols, sep=',', want_order=False):
    """OpenFace CSV -> 초당 평균 (S, len(cols)). 실패시 None."""
    want = set(['timestamp', 'success'] + cols)
    try:
        df = pd.read_csv(path, sep=sep,
                         usecols=lambda c: c.strip() in want)
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    if 'timestamp' not in df.columns:
        return None
    have = [c for c in cols if c in df.columns]
    if len(have) < len(cols) * 0.6:
        return None
    if 'success' in df.columns:
        df = df[df['success'] == 1]
    if len(df) < 80:
        return None
    df = df.copy()
    # 일부 코퍼스(LMVD)는 컬럼에 문자열이 섞여 object dtype 이 된다 -> 강제 수치화
    for c in have + ['timestamp']:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    if len(df) < 80:
        return None
    df['sec'] = np.floor(df['timestamp']).astype(int)
    out = df.groupby('sec')[have].mean().values
    out = np.nan_to_num(out.astype(np.float64))
    if want_order:
        od = _lmk_order(have)
        if len(od) != 136:
            return None
        out = out[:, od]
    return out


# ------------------------------------------------------------------- LMVD
def _lmvd_label(n):
    if 1 <= n <= 601 or 1117 <= n <= 1423:
        return 1
    if 602 <= n <= 1116 or 1425 <= n <= 1824:
        return 0
    return None


def load_lmvd_lmk():
    """LMVD 를 68 landmark 입력으로. acoustic 은 동일(VGGish 128)."""
    R = ROOT / 'LMVD' / 'extracted'
    Xv, Xa, y = [], [], []
    for f in sorted((R / 'Video_feature').glob('*.csv')):
        m = re.search(r'(\d+)', f.stem)
        if not m:
            continue
        lab = _lmvd_label(int(m.group(1)))
        if lab is None:
            continue
        fa = R / 'Audio_feature' / f'{f.stem}.npy'
        if not fa.exists():
            continue
        V = _persec(f, LMK, want_order=True)
        if V is None or len(V) < 40:
            continue
        try:
            A = np.nan_to_num(np.load(fa).astype(np.float64))
        except Exception:
            continue
        if A.ndim != 2 or len(A) < 10:
            continue
        Xv.append(resamp(zs(norm_face(V))))
        Xa.append(resamp(zs(A)))
        y.append(lab)
    return dict(Xv=np.asarray(Xv, np.float32), Xa=np.asarray(Xa, np.float32),
                y=np.asarray(y), fold=None)


def load_cmdc_lmk():
    """CMDC 를 68 landmark 입력으로. acoustic 은 동일(질문별 128-d)."""
    import pickle
    Xv, Xa, y = [], [], []
    for d in sorted(glob.glob(str(ROOT / 'CMDC' / 'extracted') + '/*/')):
        name = os.path.basename(d.rstrip('/'))
        if not (name.startswith('HC') or name.startswith('MDD')):
            continue
        segs, auds = [], []
        for q in range(1, 13):
            fv = Path(d) / f'Q{q}.csv'
            if not fv.exists():
                continue
            s2 = _persec(fv, LMK, want_order=True)
            if s2 is None:
                continue
            segs.append(s2)
            fa = Path(d) / f'Q{q}.pkl'
            if fa.exists():
                try:
                    v = np.asarray(pickle.load(open(fa, 'rb')), dtype=np.float64).reshape(-1)
                    auds.append(v)
                except Exception:
                    pass
        if not segs or len(auds) < 3:
            continue
        V = np.vstack(segs)
        if len(V) < 40:
            continue
        A = np.vstack([a[None, :] for a in auds])
        Xv.append(resamp(zs(norm_face(V))))
        Xa.append(resamp(zs(A)))
        y.append(1 if name.startswith('MDD') else 0)
    return dict(Xv=np.asarray(Xv, np.float32), Xa=np.asarray(Xa, np.float32),
                y=np.asarray(y), fold=None)


def load_lmvd():
    """visual = OpenFace AU+gaze+pose (초당), acoustic = VGGish 128 (.npy).
    공개 논문 명세("AUs, eye landmarks, gaze, head pose" + "VGGish 128")와 동일 계열."""
    R = ROOT / 'LMVD' / 'extracted'
    cols = AUS + GAZE + POSE
    Xv, Xa, y = [], [], []
    for f in sorted((R / 'Video_feature').glob('*.csv')):
        m = re.search(r'(\d+)', f.stem)
        if not m:
            continue
        lab = _lmvd_label(int(m.group(1)))
        if lab is None:
            continue
        fa = R / 'Audio_feature' / f'{f.stem}.npy'
        if not fa.exists():
            continue
        V = _persec(f, cols)
        if V is None or len(V) < 40:
            continue
        try:
            A = np.nan_to_num(np.load(fa).astype(np.float64))
        except Exception:
            continue
        if A.ndim != 2 or len(A) < 10:
            continue
        Xv.append(resamp(zs(V)))
        Xa.append(resamp(zs(A)))
        y.append(lab)
    return dict(Xv=np.asarray(Xv, np.float32), Xa=np.asarray(Xa, np.float32),
                y=np.asarray(y), fold=None)


# ------------------------------------------------------------------- CMDC
def load_cmdc():
    """visual = Q1..Q12 OpenFace AU 를 시간순 concat (초당).
    acoustic = 질문별 Q*.pkl (1,128) 을 질문 순서대로 쌓은 (Q,128) 시퀀스.
    주의: CMDC 오디오는 질문당 1벡터라 시간해상도가 낮다. 전 모델이 같은 입력을 받으므로 비교는 공정."""
    Xv, Xa, y = [], [], []
    for d in sorted(glob.glob(str(ROOT / 'CMDC' / 'extracted') + '/*/')):
        name = os.path.basename(d.rstrip('/'))
        if not (name.startswith('HC') or name.startswith('MDD')):
            continue
        segs, auds = [], []
        for q in range(1, 13):
            fv = Path(d) / f'Q{q}.csv'
            if not fv.exists():
                continue
            s = _persec(fv, AUS)
            if s is None:
                continue
            segs.append(s)
            fa = Path(d) / f'Q{q}.pkl'
            if fa.exists():
                try:
                    import pickle
                    v = np.asarray(pickle.load(open(fa, 'rb')), dtype=np.float64).reshape(-1)
                    auds.append(v)
                except Exception:
                    pass
        if not segs or len(auds) < 3:
            continue
        V = np.vstack(segs)
        if len(V) < 40:
            continue
        A = np.vstack([a[None, :] for a in auds])
        Xv.append(resamp(zs(V)))
        Xa.append(resamp(zs(A)))
        y.append(1 if name.startswith('MDD') else 0)
    return dict(Xv=np.asarray(Xv, np.float32), Xa=np.asarray(Xa, np.float32),
                y=np.asarray(y), fold=None)


# ------------------------------------------------------------------ E-DAIC
def load_edaic(official_split=True):
    """visual = OpenFace AU+gaze+pose (초당), acoustic = OpenSMILE eGeMAPS (';' 구분, 초당).
    AVEC2019 공식 train/dev/test 분할 사용."""
    lab, spl = {}, {}
    for spn, fold in [('train', 'train'), ('dev', 'valid'), ('test', 'test')]:
        p = ROOT / 'E-DAIC' / 'labels' / f'{spn}_split.csv'
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            pid = r.get('Participant_ID', '').strip()
            b = r.get('PHQ_Binary', '').strip()
            if pid and b in ('0', '1'):
                lab[pid] = int(b); spl[pid] = fold
    cols = AUS + GAZE + POSE
    Xv, Xa, y, fold = [], [], [], []
    for d in sorted(glob.glob(str(ROOT / 'E-DAIC' / 'extracted') + '/*_P/')):
        pid = os.path.basename(d.rstrip('/')).replace('_P', '')
        if pid not in lab:
            continue
        fv = Path(d) / 'features' / f'{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'
        fa = Path(d) / 'features' / f'{pid}_OpenSMILE2.3.0_egemaps.csv'
        if not (fv.exists() and fa.exists()):
            continue
        V = _persec(fv, cols)
        if V is None or len(V) < 40:
            continue
        try:
            A = pd.read_csv(fa, sep=';')
            A = A.drop(columns=[c for c in ['name', 'frameTime'] if c in A.columns])
            A = np.nan_to_num(A.values.astype(np.float64))
        except Exception:
            continue
        if A.ndim != 2 or len(A) < 10:
            continue
        # 초당 축약 (eGeMAPS 는 10ms 프레임 -> 100프레임/초)
        k = max(1, len(A) // max(1, len(V)))
        A = A[:len(A) // k * k].reshape(-1, k, A.shape[1]).mean(1)
        Xv.append(resamp(zs(V)))
        Xa.append(resamp(zs(A)))
        y.append(lab[pid]); fold.append(spl[pid])
    fd = np.asarray(fold) if official_split else None
    return dict(Xv=np.asarray(Xv, np.float32), Xa=np.asarray(Xa, np.float32),
                y=np.asarray(y), fold=fd)
