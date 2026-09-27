"""bench/repeat_cv.py — 반복 층화 교차검증 + 쌍대 신뢰구간.

왜 필요한가:
  공식 분할은 시험셋이 작다(MUD3 66명, E-DAIC 54명). 이 규모에서는 bootstrap
  신뢰구간이 0을 포함해 "유의하다"고 쓸 수 없다.
  전체 표본을 5겹 x R반복으로 돌리면 **모든 표본이 한 번씩 시험에 쓰이므로**
  검정력이 올라간다.

보고하는 것:
  1) 반복 간 대응 검정 — 모델 간 차이가 반복에 걸쳐 일관된가
  2) 표본 bootstrap 신뢰구간 — 다른 사람을 뽑아도 같은 결론인가
  둘 다 보고해야 "일관성"과 "일반화"를 분리해 말할 수 있다.

사용:
  python -m bench.repeat_cv --corpus mud3 --a 'depdetector+full' --b depdetector
"""
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats

from bench.data import LOADERS
from bench.run import fit_eval, prep_covs, needs_covs


def oof(model_name, d, repeats=5, pca=20):
    """5겹 x repeats 반복 -> (repeats, N) out-of-fold 예측 확률."""
    y = d['y']
    N = len(y)
    out = np.zeros((repeats, N))
    for r in range(repeats):
        for tri, tei in StratifiedKFold(5, shuffle=True, random_state=r).split(np.zeros(N), y):
            rng = np.random.RandomState(r)
            tri = tri.copy()
            rng.shuffle(tri)
            k = max(1, int(len(tri) * 0.15))
            tr = np.zeros(N, bool)
            va = np.zeros(N, bool)
            te = np.zeros(N, bool)
            va[tri[:k]] = True
            tr[tri[k:]] = True
            te[tei] = True
            c = w = None
            if needs_covs(model_name):
                c, w = prep_covs(d, tr, pca)
            p, ite, _, _, _ = fit_eval(model_name, d, tr, va, te, r, covs=c, wfeat=w)
            out[r, ite] = p
        print(f'    반복 {r + 1}/{repeats} 완료', flush=True)
    return out


def boot_diff(y, pa, pb, n=2000, seed=0):
    """같은 부트스트랩 표본에서 두 모델 차이 -> 쌍대 비교."""
    rng = np.random.RandomState(seed)
    N = len(y)
    df, da = [], []
    for _ in range(n):
        i = rng.randint(0, N, N)
        if len(np.unique(y[i])) < 2:
            continue
        df.append(f1_score(y[i], (pa[i] > .5).astype(int), zero_division=0)
                  - f1_score(y[i], (pb[i] > .5).astype(int), zero_division=0))
        da.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return np.array(df), np.array(da)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--models', required=True,
                    help='쉼표 구분. 첫 번째가 기준(baseline), 나머지를 그것과 비교한다')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--pca', type=int, default=20)
    ap.add_argument('--out', default='')
    g = ap.parse_args()

    names = [x.strip() for x in g.models.split(',') if x.strip()]
    d = LOADERS[g.corpus]()
    y = d['y']
    print(f'[{g.corpus}] n={len(y)} 우울={int(y.sum())}  |  5겹 x {g.repeats}반복  |  모델 {len(names)}개',
          flush=True)

    P = {}
    for nm in names:
        print(f'  >> {nm}', flush=True)
        P[nm] = oof(nm, d, g.repeats, g.pca)

    # 개별 성적
    print()
    print(f'{"모델":<26}{"F1":>18}{"AUC":>18}')
    S = {}
    for nm in names:
        f = [f1_score(y, (p > .5).astype(int), zero_division=0) for p in P[nm]]
        a = [roc_auc_score(y, p) for p in P[nm]]
        S[nm] = (f, a)
        print(f'{nm:<26}{np.mean(f):>10.4f}±{np.std(f):<7.4f}{np.mean(a):>10.4f}±{np.std(a):<7.4f}')

    # 기준 대비 비교
    base = names[0]
    print()
    print(f'기준: {base}')
    rows = []
    for nm in names[1:]:
        line = [nm]
        for k, i in [('F1', 0), ('AUC', 1)]:
            x, z = S[nm][i], S[base][i]
            t, pv = stats.ttest_rel(x, z)
            line.append(f'{np.mean(x) - np.mean(z):+.4f} (p={pv:.3f})')
        df, da = boot_diff(y, P[nm].mean(0), P[base].mean(0))
        for k, v in [('F1', df), ('AUC', da)]:
            lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
            line.append(f'[{lo:+.3f},{hi:+.3f}]' + ('*' if (lo > 0 or hi < 0) else ''))
        rows.append(line)
        print(f'  {nm:<26} F1 {line[1]:<20} AUC {line[2]:<20} CI_F1 {line[3]:<18} CI_AUC {line[4]}')
    print()
    print('  * = 신뢰구간이 0을 제외 (유의)')

    if g.out:
        import csv as _csv
        with open(g.out, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(['corpus', 'model', 'f1_mean', 'f1_std', 'auc_mean', 'auc_std'])
            for nm in names:
                f, a = S[nm]
                w.writerow([g.corpus, nm, f'{np.mean(f):.4f}', f'{np.std(f):.4f}',
                            f'{np.mean(a):.4f}', f'{np.std(a):.4f}'])
        print(f'저장: {g.out}')


if __name__ == '__main__':
    main()
