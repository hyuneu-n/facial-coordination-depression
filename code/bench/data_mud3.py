"""bench/data_mud3.py — MUD3 로더 (raw / aligned 두 조건).

구조: dep_feat.pkl / nondep_feat.pkl = {'features': [유저별 [영상별 (S,161) ndarray]], 'name': [...]}
      161 = 앞 136 (68 landmark x2, visual) + 뒤 25 (acoustic) — D-Vlog 과 동일 규격.
      labels.csv = id, labels, names, split(train/val/test)

유저 단위 샘플: 유저의 모든 영상을 시간순 concat -> (T,161) -> visual/acoustic 분리 -> T=256 리샘플.
베이스라인들이 받는 것과 동일한 평탄화(flat) 처리다. 계층 구조는 별도 실험(exp135)에서 다룬다.

aligned=True: Procrustes(Kabsch) 정렬로 위치·스케일·회전을 제거한 landmark 사용.
  -> confound(촬영조건)를 걷어낸 조건. raw 와 짝지어 보면 각 모델이 confound에
     얼마나 의존하는지가 드러난다.
"""
import numpy as np, pickle, csv

from bench.data import ROOT, T, resamp, zs

NV = 136  # visual dims (68 landmark x 2)


def _kabsch_align(P):
    """(F,68,2) -> 평균 형상에 Procrustes 정렬 (위치·스케일·회전 제거)."""
    P = P - P.mean(1, keepdims=True)
    sc = np.sqrt((P ** 2).sum(2).mean(1, keepdims=True))
    sc = np.where(sc < 1e-8, 1.0, sc)
    P = P / sc[:, None]
    ref = P.mean(0)
    rn = np.linalg.norm(ref)
    if rn > 1e-8:
        ref = ref / rn * np.sqrt(ref.shape[0])
    out = np.empty_like(P)
    for i in range(P.shape[0]):
        H = P[i].T @ ref
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, d]) @ U.T
        out[i] = P[i] @ R.T
    return out


def load_mud3(aligned=False, normalize=True):
    """normalize=False 면 z-score 없이 원시 좌표 그대로 쓴다(진짜 raw).
    z-score 는 좌표별 평균·스케일을 지우므로 위치·크기 confound 를 이미 상당 부분 제거한다.
    confound 의존도를 재려면 그 제거가 없는 조건이 필요하다."""
    lab, spl = {}, {}
    for r in csv.DictReader(open(ROOT / 'MUD3' / 'raw' / 'labels.csv')):
        n = r['names'].strip()
        lab[n] = int(r['labels'])
        s = r['split'].strip().lower()
        spl[n] = 'valid' if s.startswith('val') else s

    Xv, Xa, y, fold = [], [], [], []
    for fn in ['dep_feat.pkl', 'nondep_feat.pkl']:
        D = pickle.load(open(ROOT / 'MUD3' / 'raw' / fn, 'rb'))
        for name, vids in zip(D['name'], D['features']):
            if name not in lab:
                continue
            segs = [np.asarray(v, dtype=np.float64) for v in vids
                    if np.asarray(v).ndim == 2 and np.asarray(v).shape[1] >= NV + 1]
            if not segs:
                continue
            M = np.nan_to_num(np.vstack(segs))
            if len(M) < 8:
                continue
            V, A = M[:, :NV], M[:, NV:]
            if aligned:
                V = _kabsch_align(V.reshape(-1, 68, 2)).reshape(-1, NV)
            Xv.append(resamp(zs(V) if normalize else V))
            Xa.append(resamp(zs(A) if normalize else A))
            y.append(lab[name]); fold.append(spl[name])
    return dict(Xv=np.asarray(Xv, np.float32), Xa=np.asarray(Xa, np.float32),
                y=np.asarray(y), fold=np.asarray(fold))
