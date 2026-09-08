"""exp138 — DISFA 실제 AU intensity로 AU15-AU17 협응(coupling) 검증.
D-Vlog에서 landmark기반 협응으로 찾은 판별부위=입·턱(AU15,17 해당)을 실제 AU라벨로 뒷받침.
27명 전체, 프레임별 AU 강도(0~5) 상관 + 협응 구조(다른 AU쌍과 비교) 확인.
"""
import os, glob, numpy as np, itertools
from scipy import stats
D = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/DISFA/ActionUnit_Labels')
subs = sorted(os.listdir(D))
print(f'피험자 {len(subs)}명', flush=True)

def load_au(sub, au):
    p = f'{D}/{sub}/{sub}_au{au}.txt'
    if not os.path.exists(p): return None
    vals = np.array([int(l.split(',')[1]) for l in open(p) if l.strip()])
    return vals

AUS = [1,2,4,5,6,9,12,15,17,20,25,26]
# ---- 전체 피험자 pooled correlation matrix (프레임 단위) ----
all_r = {tuple(sorted(p)): [] for p in itertools.combinations(AUS,2)}
per_subj_r1517 = []
for sub in subs:
    data = {au: load_au(sub, au) for au in AUS}
    if any(v is None for v in data.values()): continue
    L = min(len(v) for v in data.values())
    for a,b in itertools.combinations(AUS,2):
        va, vb = data[a][:L], data[b][:L]
        if va.std()>0 and vb.std()>0:
            r,_ = stats.pearsonr(va, vb)
            all_r[tuple(sorted((a,b)))].append(r)
    if data[15][:L].std()>0 and data[17][:L].std()>0:
        r,_ = stats.pearsonr(data[15][:L], data[17][:L])
        per_subj_r1517.append(r)

print('\n[전체 AU쌍 평균 상관 — 내림차순 top 15]', flush=True)
avg_r = {k: np.mean(v) for k,v in all_r.items() if v}
for (a,b), r in sorted(avg_r.items(), key=lambda x:-x[1])[:15]:
    mark = '  <<< AU15-AU17' if {a,b}=={15,17} else ''
    print(f'  AU{a}-AU{b}: r={r:.3f}  (n={len(all_r[(a,b)])}명){mark}', flush=True)

print(f'\n[AU15-AU17 특정] 피험자별 상관 평균={np.mean(per_subj_r1517):.3f} ± {np.std(per_subj_r1517):.3f} (n={len(per_subj_r1517)}명)', flush=True)
print(f'  전체 AU쌍 상관 중 순위: {sorted(avg_r.values(), reverse=True).index(avg_r[(15,17)])+1} / {len(avg_r)}', flush=True)

# t-test: AU15-17 상관이 다른 쌍들 평균보다 높은지
others = [v for k,v in avg_r.items() if k != (15,17)]
t,p = stats.ttest_1samp(others, avg_r[(15,17)])
print(f'  AU15-17({avg_r[(15,17)]:.3f}) vs 나머지 66쌍 평균({np.mean(others):.3f}): one-sample t={t:.3f} p={p:.5f}', flush=True)
print('DONE', flush=True)
