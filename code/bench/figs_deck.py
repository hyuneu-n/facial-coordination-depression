"""발표용 그림 2장.
 fig20: coordination 모듈 이식 전후 (MUD3 두 조건 x 두 베이스라인)
 fig21: MUD3 전 모델 순위 — 이식 모델이 최상단
"""
import csv, os
import numpy as np
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
RES = '/home/hyuneun/disk_b/🟡facial-prodrome/results/bench'

NAVY, BLUE, RED, GREY = '#002852', '#0066A5', '#b91c1c', '#9aa3af'


def read(c):
    p = f'{RES}/{c}.csv'
    if not os.path.exists(p):
        return {}
    return {r['model']: r for r in csv.DictReader(open(p))}


# ---------------------------------------------------------------- fig20
def fig20():
    conds = [('mud3', 'MUD3\n(원본)'), ('mud3_aligned', 'MUD3\n(촬영조건 제거)')]
    pairs = [('depdetector', 'depcoord', 'DepDetector'),
             ('blstm', 'blstmcoord', 'BLSTM')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (metric, mname) in zip(axes, [('f1_ens', 'F1'), ('auc_ens', 'AUC')]):
        labels, base, graft = [], [], []
        for c, cn in conds:
            D = read(c)
            for b, g, bn in pairs:
                if b in D and g in D:
                    labels.append(f'{bn}\n{cn}')
                    base.append(float(D[b][metric]))
                    graft.append(float(D[g][metric]))
        x = np.arange(len(labels)); w = 0.36
        ax.bar(x - w / 2, base, w, label='기존 모델', color=GREY, edgecolor='k', linewidth=.5)
        ax.bar(x + w / 2, graft, w, label='+ coordination', color=BLUE, edgecolor='k', linewidth=.5)
        for i, (b, g) in enumerate(zip(base, graft)):
            ax.text(i + w / 2, g + .006, f'+{(g-b)*100:.1f}%p', ha='center',
                    fontsize=10.5, fontweight='bold', color=BLUE if g > b else RED)
            ax.text(i - w / 2, b + .006, f'{b:.3f}', ha='center', fontsize=9.5, color='#555')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(mname, fontsize=12)
        ax.set_ylim(min(base + graft) - .06, max(base + graft) + .05)
        ax.set_title(f'{mname} — 이식 전후', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=.3); ax.set_axisbelow(True); ax.legend(fontsize=10.5)
    fig.suptitle('coordination 인코더를 붙이면 두 모델 모두 성능이 오른다',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{RES}/fig20_graft_effect.png', dpi=150)
    print('saved fig20')


# ---------------------------------------------------------------- fig21
NAME = {'depcoord': 'DepDetector + coordination  (제안)',
        'blstmcoord': 'BLSTM + coordination  (제안)',
        'depdetector': 'DepDetector (AAAI 2022)',
        'blstm': 'BLSTM (2019)', 'tfn': 'TFN (2017)',
        'tamfn': 'TAMFN (TNSRE 2023)', 'ours': 'coordination 단독 (제안)'}


def fig21():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (c, t) in zip(axes, [('mud3', 'MUD3 (원본)'),
                                 ('mud3_aligned', 'MUD3 (촬영조건 제거)')]):
        D = read(c)
        rows = [(NAME[m], float(r['f1_ens']), m) for m, r in D.items() if m in NAME]
        rows.sort(key=lambda z: z[1])
        y = np.arange(len(rows))
        cols = [BLUE if ('coord' in m or m == 'ours') else GREY for _, _, m in rows]
        ax.barh(y, [v for _, v, _ in rows], color=cols, edgecolor='k', linewidth=.5)
        for i, (_, v, _) in enumerate(rows):
            ax.text(v + .004, i, f'{v:.3f}', va='center', fontsize=10.5, fontweight='bold')
        ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in rows], fontsize=10.5)
        ax.set_xlim(min(v for _, v, _ in rows) - .05, max(v for _, v, _ in rows) + .045)
        ax.set_xlabel('F1', fontsize=12)
        ax.set_title(t, fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=.3); ax.set_axisbelow(True)
    fig.suptitle('파란색 = coordination을 사용한 모델', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{RES}/fig21_mud3_ranking.png', dpi=150)
    print('saved fig21')


if __name__ == '__main__':
    fig20(); fig21()
