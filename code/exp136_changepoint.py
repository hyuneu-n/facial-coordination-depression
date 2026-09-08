"""exp136 — MUD3 within-user change-point 탐지. 진짜 goal(시간축 전조) 첫 시도.
입력: mud3_aligned.pkl (위치·크기·회전 제거된 표정 표현) + 오디오는 유저내 표준화(confound 제거).
방법: ruptures Pelt(rbf) — 유저별 스트림에서 change-point 1개(가장 강한 것) 탐지.
검증축:
 Q1. change-point 존재율(=유의미한 분할이 있는가)이 dep/non 다른가
 Q2. change-point 위치(정규화 0~1)가 dep/non 다르게 분포하는가 (KS test)
 Q3. 위치가 균일분포인지(confound 없음) vs 특정구간 쏠림(구조적 신호)
 Q4. change 전/후 크기(효과크기)가 dep/non 다른가
"""
import os, pickle, numpy as np, pandas as pd
import ruptures as rpt
from scipy import stats
CACHE = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_aligned.pkl')
C = pickle.load(open(CACHE, 'rb'))
MINK = 30   # 최소 영상 수 (충분한 시간해상도)

def zs(a):
    sd = a.std(0); return (a - a.mean(0)) / np.where(sd < 1e-8, 1.0, sd)

rows = []
for n, v in C.items():
    M = v['mean'].astype(np.float64)     # [K,20] pose-free 표정
    A = v['aud'].astype(np.float64)      # [K,50] raw acoustic 요약
    K = len(M)
    if K < MINK: continue
    Mz = zs(M); Az = zs(A)               # 유저내 표준화 (confound 제거)
    X = np.concatenate([Mz, Az], axis=1) # [K,70] 멀티모달 궤적
    algo = rpt.Pelt(model='rbf', min_size=max(5, K//10), jump=1).fit(X)
    try:
        bkps = algo.predict(pen=10)      # 페널티로 개수 자동 결정
    except Exception:
        bkps = [K]
    bkps = [b for b in bkps if b < K]    # 마지막(=K)은 종료지점이므로 제외
    has_cp = len(bkps) > 0
    pos = bkps[0]/K if has_cp else np.nan     # 가장 이른 change-point 정규화 위치
    # 효과크기: 그 지점 전후 표정 벡터 평균 차이
    eff = np.nan
    if has_cp:
        b = bkps[0]
        pre, post = Mz[:b], Mz[b:]
        if len(pre) >= 3 and len(post) >= 3:
            eff = np.linalg.norm(post.mean(0) - pre.mean(0))
    rows.append(dict(name=n, y=v['label'], split=v['split'], K=K,
                      n_cp=len(bkps), has_cp=has_cp, cp_pos=pos, cp_eff=eff))
df = pd.DataFrame(rows)
print(f'분석대상(K>={MINK}) = {len(df)}명 (dep {int(df.y.sum())})', flush=True)

d = df[df.split != 'test']
print('\n[Q1] change-point 존재율', flush=True)
for g,tag in [(0,'비우울'),(1,'우울')]:
    sub = d[d.y==g]
    print(f'  {tag}: {sub.has_cp.mean()*100:.1f}% ({sub.has_cp.sum()}/{len(sub)})  n_cp 평균={sub.n_cp.mean():.2f}', flush=True)
ct = pd.crosstab(d.y, d.has_cp)
chi2,p,_,_ = stats.chi2_contingency(ct)
print(f'  카이제곱 검정 p={p:.5f}', flush=True)

print('\n[Q2] change-point 위치 분포 (KS test, has_cp만)', flush=True)
pos0 = d[(d.y==0)&d.has_cp].cp_pos.dropna(); pos1 = d[(d.y==1)&d.has_cp].cp_pos.dropna()
print(f'  비우울 위치평균={pos0.mean():.3f}  우울 위치평균={pos1.mean():.3f}', flush=True)
ks,p = stats.ks_2samp(pos0,pos1); print(f'  KS test p={p:.5f}', flush=True)

print('\n[Q3] 위치가 균일분포(0~1)인지 — Kolmogorov-Smirnov vs Uniform', flush=True)
for g,tag,pos in [(0,'비우울',pos0),(1,'우울',pos1)]:
    ks,p = stats.kstest(pos, 'uniform')
    print(f'  {tag} vs Uniform(0,1): KS p={p:.5f}  {"쏠림 있음" if p<0.05 else "균일(구조없음)"}', flush=True)
# 히스토그램 (5구간)
for g,tag,pos in [(0,'비우울',pos0),(1,'우울',pos1)]:
    h,_ = np.histogram(pos, bins=5, range=(0,1))
    print(f'  {tag} 위치 히스토그램(5구간): {h.tolist()}', flush=True)

print('\n[Q4] change 전후 효과크기', flush=True)
e0 = d[(d.y==0)&d.has_cp].cp_eff.dropna(); e1 = d[(d.y==1)&d.has_cp].cp_eff.dropna()
t,p = stats.ttest_ind(e0,e1,equal_var=False)
print(f'  비우울 eff={e0.mean():.4f}  우울 eff={e1.mean():.4f}  t={t:.3f} p={p:.5f}', flush=True)

print('\n[confound 점검] change-point 존재/위치가 영상수(K)와 상관있는가', flush=True)
print(f'  has_cp vs K: 상관 point-biserial r={np.corrcoef(d.has_cp.astype(int), d.K)[0,1]:+.3f}', flush=True)
print(f'  cp_pos vs K: r={np.corrcoef(d[d.has_cp].cp_pos, d[d.has_cp].K)[0,1]:+.3f}', flush=True)
print('DONE', flush=True)
