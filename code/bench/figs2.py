"""벤치마크 2차 결과 그림 3종.
 fig22: 코퍼스별 host-범용성 (이식효과 히트맵)
 fig23: 모듈 ablation (반복 교차검증)
 fig24: 앙상블 vs 시드평균 — 측정 방식이 결론을 바꾼다
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for f in ['NanumBarunGothic', 'NanumGothic', 'DejaVu Sans']:
    try:
        plt.rcParams['font.family'] = f
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False
R = '/home/hyuneun/disk_b/🟡facial-prodrome/results/bench'
NAVY, BLUE, RED, GREY = '#002852', '#0066A5', '#b91c1c', '#9aa3af'

CORP = [('lmvd', 'LMVD\n(n=1556)'), ('dvlog', 'D-Vlog\n(n=952)'),
        ('mud3', 'MUD3 원본\n(n=650)'), ('mud3_aligned', 'MUD3 제거\n(n=650)'),
        ('cmdc', 'CMDC\n(n=45)')]
HOST = ['blstm', 'tfn', 'depdetector', 'tamfn']
HLBL = ['BLSTM', 'TFN', 'DepDetector', 'TAMFN']


def G(c, m, k):
    p = f'{R}/{c}.csv'
    if not os.path.exists(p):
        return None
    D = {r['model']: r for r in csv.DictReader(open(p))}
    for cand in (m + '_h', m + '_abl', m):
        if cand in D:
            return float(D[cand][k])
    return None


def fig22():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, (key, t) in zip(axes, [('f1_mean', 'F1'), ('auc_mean', 'AUC')]):
        M = np.full((len(CORP), len(HOST)), np.nan)
        for i, (c, _) in enumerate(CORP):
            for j, h in enumerate(HOST):
                b, g = G(c, h, key), G(c, h + '+full', key)
                if b is not None and g is not None:
                    M[i, j] = (g - b) * 100
        v = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap='RdBu', vmin=-v, vmax=v, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:+.1f}', ha='center', va='center',
                            fontsize=13, fontweight='bold',
                            color='white' if abs(M[i, j]) > v * .55 else '#222')
        ax.set_xticks(range(len(HOST))); ax.set_xticklabels(HLBL, fontsize=11)
        ax.set_yticks(range(len(CORP))); ax.set_yticklabels([n for _, n in CORP], fontsize=11)
        ax.set_title(f'{t} 변화 (%p)', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=.04)
    fig.suptitle('coordination 모듈 이식 효과 — 파란색이 상승, 빨간색이 하락\n'
                 'LMVD 만 8칸 모두 하락이 없다', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{R}/fig22_graft_grid.png', dpi=150)
    print('saved fig22')


def fig23():
    LBL = {'': '기존 모델', '+full': '+제안(전체)', '+nospd': '+곡면기하 제거',
           '+notime': '+시간모델 제거', '+nocov': '+coordination 제거'}
    order = ['', '+full', '+nospd', '+notime', '+nocov']
    cols = [GREY, BLUE, '#7fb3d5', '#7fb3d5', RED]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (c, t) in zip(axes, [('mud3', 'MUD3 원본'), ('mud3_aligned', 'MUD3 촬영조건 제거'),
                                 ('dvlog', 'D-Vlog')]):
        p = f'{R}/ablcv_{c}_depdetector.csv'
        if not os.path.exists(p):
            ax.set_visible(False); continue
        D = {r['model']: float(r['f1_mean']) for r in csv.DictReader(open(p))}
        S = {r['model']: float(r['f1_std']) for r in csv.DictReader(open(p))}
        keys = ['depdetector' + o for o in order]
        vals = [D.get(k, np.nan) for k in keys]
        errs = [S.get(k, 0) for k in keys]
        ax.bar(range(5), vals, yerr=errs, color=cols, edgecolor='k', linewidth=.6, capsize=4)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v + .012, f'{v:.3f}', ha='center', fontsize=10.5, fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels([LBL[o] for o in order], fontsize=9.5, rotation=18, ha='right')
        ax.set_ylabel('F1'); ax.set_title(t, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=.3); ax.set_axisbelow(True)
        lo = np.nanmin(vals); ax.set_ylim(max(0, lo - .09), np.nanmax(vals) + .06)
    fig.suptitle('설계 요소별 기여 (DepDetector 기준, 반복 교차검증)\n'
                 '곡면기하를 빼도 떨어지지 않는다', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{R}/fig23_ablation_cv.png', dpi=150)
    print('saved fig23')


def fig24():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    labs, ens, mean = [], [], []
    for c, n in [('dvlog', 'D-Vlog'), ('mud3', 'MUD3')]:
        for h, hl in zip(HOST, HLBL):
            be, ge = G(c, h, 'f1_ens'), G(c, h + '+full', 'f1_ens')
            bm, gm = G(c, h, 'f1_mean'), G(c, h + '+full', 'f1_mean')
            if None in (be, ge, bm, gm):
                continue
            labs.append(f'{n}\n{hl}')
            ens.append((ge - be) * 100)
            mean.append((gm - bm) * 100)
    x = np.arange(len(labs)); w = .38
    ax.bar(x - w / 2, ens, w, label='앙상블 기준', color=BLUE, edgecolor='k', linewidth=.5)
    ax.bar(x + w / 2, mean, w, label='시드평균 기준', color=GREY, edgecolor='k', linewidth=.5)
    ax.axhline(0, color='k', lw=1)
    for i, (a, b) in enumerate(zip(ens, mean)):
        if a > 0 > b:
            ax.text(i, max(a, 0) + .7, '부호 역전', ha='center', fontsize=10,
                    fontweight='bold', color=RED)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=10)
    ax.set_ylabel('이식 효과 (%p)')
    ax.set_title('측정 방식이 결론을 바꾼다 — 앙상블은 분산이 큰 모델을 유리하게 만든다',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f'{R}/fig24_ensemble_artifact.png', dpi=150)
    print('saved fig24')


if __name__ == '__main__':
    fig22(); fig23(); fig24()
