"""bench/figs.py — 비교 그림 2종.
  fig18: 파라미터 대비 성능 (좌상단이 좋음)
  fig19: 조기탐지 — 사용 구간 비율별 AUC (선이 평평할수록 일찍 잡는다)
"""
import csv, re, numpy as np, time, torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for fam in ['NanumBarunGothic', 'NanumGothic', 'DejaVu Sans']:
    try:
        plt.rcParams['font.family'] = fam
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

OUT = Path('/home/hyuneun/disk_b/🟡facial-prodrome/results/bench')
NAME = {'blstm': 'BLSTM', 'tfn': 'TFN', 'depdetector': 'DepDetector',
        'tamfn': 'TAMFN', 'ours': '제안 모델'}
COL = {'blstm': '#888888', 'tfn': '#7fb3d5', 'depdetector': '#f0932b',
       'tamfn': '#b8a0d0', 'ours': '#c0392b'}


def read(corpus):
    f = OUT / f'{corpus}.csv'
    if not f.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        m = r['model']
        if m not in NAME:      # 태그 붙은 변형은 제외
            continue
        out[m] = r
    return out


def fig_params(corpora=('dvlog', 'lmvd', 'cmdc', 'edaic')):
    fig, axes = plt.subplots(1, len(corpora), figsize=(4.2 * len(corpora), 4.2))
    if len(corpora) == 1:
        axes = [axes]
    for ax, c in zip(axes, corpora):
        D = read(c)
        if not D:
            ax.set_visible(False); continue
        for m, r in D.items():
            x = int(r['params']); y = float(r['f1_ens'])
            ax.scatter(x, y, s=190 if m == 'ours' else 110, c=COL[m],
                       marker='*' if m == 'ours' else 'o',
                       edgecolors='k', linewidths=0.6, zorder=3)
            ax.annotate(NAME[m], (x, y), textcoords='offset points',
                        xytext=(7, 5), fontsize=9)
        ax.set_xscale('log')
        ax.set_xlabel('파라미터 수 (log)')
        ax.set_ylabel('F1 (앙상블)')
        ax.set_title(c.upper())
        ax.grid(alpha=0.3)
    fig.suptitle('파라미터 대비 성능 — 좌상단일수록 작고 좋다', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'fig18_params_vs_f1.png', dpi=150)
    print('saved fig18')


def fig_early(corpus='dvlog'):
    f = OUT / f'early_{corpus}.csv'
    if not f.exists():
        print('no early csv'); return
    D = {}
    for r in csv.DictReader(open(f)):
        m = re.sub(r'_(fid_)?e2?$', '', r['model'])
        if m not in NAME:
            continue
        D.setdefault(m, []).append((float(r['frac']), float(r['auc_mean']),
                                    float(r['auc_std'])))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for m, v in D.items():
        v.sort()
        x = np.array([a * 100 for a, _, _ in v])
        y = np.array([b for _, b, _ in v]); e = np.array([c for _, _, c in v])
        lw = 2.6 if m == 'ours' else 1.6
        axes[0].errorbar(x, y, yerr=e, marker='o', lw=lw, color=COL[m],
                         label=NAME[m], capsize=3)
        axes[1].plot(x, y / y[-1] * 100, marker='o', lw=lw, color=COL[m],
                     label=NAME[m])
    axes[0].set_xlabel('사용한 세션 앞부분 비율 (%)'); axes[0].set_ylabel('AUC')
    axes[0].set_title('절대 성능'); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].axhline(100, color='k', ls=':', lw=1)
    axes[1].set_xlabel('사용한 세션 앞부분 비율 (%)')
    axes[1].set_ylabel('전체 사용 대비 성능 유지율 (%)')
    axes[1].set_title('유지율 — 높고 평평할수록 일찍 잡는다')
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(f'조기탐지 비교 ({corpus.upper()}) — 학습은 전체 길이, 평가만 앞부분으로 절단',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'fig19_early_vs_baselines.png', dpi=150)
    print('saved fig19')


def latency():
    from bench.models import build
    from bench.metrics import nparams
    print(f'{"모델":14s} {"파라미터":>10s} {"CPU ms":>8s}')
    for m in ['blstm', 'tfn', 'depdetector', 'tamfn']:
        net = build(m, 136, 25).eval()
        xv, xa = torch.randn(1, 256, 136), torch.randn(1, 256, 25)
        with torch.no_grad():
            for _ in range(5):
                net(xv, xa)
            t0 = time.perf_counter()
            for _ in range(50):
                net(xv, xa)
        print(f'{NAME[m]:14s} {nparams(net):>10,} {(time.perf_counter()-t0)/50*1000:>8.2f}')


if __name__ == '__main__':
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if a in ('all', 'params'):
        fig_params()
    if a in ('all', 'early'):
        fig_early()
    if a in ('all', 'lat'):
        latency()
