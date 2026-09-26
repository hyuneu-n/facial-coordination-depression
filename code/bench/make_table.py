"""bench/make_table.py — 코퍼스 x 모델 통합 비교표 (마크다운 + CSV).

공개 수치는 지형 참고로만 병기한다. D-Vlog 외에는 프로토콜이 달라 직접 비교 불가.
"""
import csv
from pathlib import Path

OUT = Path('/home/hyuneun/disk_b/🟡facial-prodrome/results/bench')

ORDER = ['blstm', 'tfn', 'depdetector', 'tamfn', 'ours']
NAME = {'blstm': 'BLSTM (2019)', 'tfn': 'TFN (2017)',
        'depdetector': 'DepDetector (AAAI 2022)', 'tamfn': 'TAMFN (TNSRE 2023)',
        'ours': '**제안 모델**'}
CORP = [('dvlog', 'D-Vlog (공식 split)'), ('lmvd', 'LMVD (5-fold CV)'),
        ('cmdc', 'CMDC (5-fold CV, n=45)'), ('edaic', 'E-DAIC (AVEC2019 공식)'),
        ('mud3', 'MUD3 raw (confound 포함)'),
        ('mud3_aligned', 'MUD3 aligned (confound 제거)')]

# CAF-Mamba (arXiv 2601.21648) Table 2 공개 F1 — 지형 참고
PUB = {'dvlog': {'blstm': .6077, 'tfn': .6155, 'depdetector': .6482, 'tamfn': .6611},
       'lmvd': {'blstm': .6783, 'tfn': .6334, 'depdetector': .6508, 'tamfn': .6984}}


def read(c):
    f = OUT / f'{c}.csv'
    if not f.exists():
        return {}
    D = {}
    for r in csv.DictReader(open(f)):
        if r['model'] in ORDER:
            D[r['model']] = r
    return D


def main():
    lines, rows = [], []
    for c, title in CORP:
        D = read(c)
        if not D:
            continue
        lines.append(f'\n### {title}\n')
        pub = PUB.get(c, {})
        head = '| 모델 | 파라미터 | F1 (seed평균) | F1 (앙상블) | AUC (seed평균) | AUC (앙상블) |'
        if pub:
            head = head + ' 공개 F1 |'
        lines.append(head)
        lines.append('|' + '---|' * (7 if pub else 6))
        best = max((float(D[m]['f1_ens']) for m in D), default=0)
        for m in ORDER:
            if m not in D:
                continue
            r = D[m]
            f1e = float(r['f1_ens'])
            mark = ' 🏆' if abs(f1e - best) < 1e-9 else ''
            cells = [NAME[m], f"{int(r['params']):,}",
                     r['f1_mean'], r['f1_ens'] + mark, r['auc_mean'], r['auc_ens']]
            if pub:
                cells.append(f"{pub[m]:.4f}" if m in pub else '—')
            lines.append('| ' + ' | '.join(cells) + ' |')
            rows.append([c, m, r['params'], r['f1_mean'], r['f1_ens'],
                         r['auc_mean'], r['auc_ens'], pub.get(m, '')])

    # confound 의존도: raw -> aligned 하락폭
    R, A = read('mud3'), read('mud3_aligned')
    if R and A:
        lines.append('\n### MUD3 confound 의존도 (raw → aligned 하락폭)\n')
        lines.append('| 모델 | raw F1 | aligned F1 | 하락 | 유지율 |')
        lines.append('|---|---|---|---|---|')
        for m in ORDER:
            if m in R and m in A:
                a, b = float(R[m]['f1_ens']), float(A[m]['f1_ens'])
                keep = b / a * 100 if a > 1e-9 else 0
                lines.append(f'| {NAME[m]} | {a:.4f} | {b:.4f} | {b-a:+.4f} | {keep:.1f}% |')
        lines.append('\n유지율이 높을수록 촬영·녹음 조건(confound)에 덜 의존한다.')

    txt = '\n'.join(lines)
    (OUT / 'TABLE_master.md').write_text(txt)
    with open(OUT / 'TABLE_master.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['corpus', 'model', 'params', 'f1_mean', 'f1_ens',
                    'auc_mean', 'auc_ens', 'published_f1'])
        w.writerows(rows)
    print(txt)
    print(f'\n저장: {OUT}/TABLE_master.md , TABLE_master.csv')


if __name__ == '__main__':
    main()
