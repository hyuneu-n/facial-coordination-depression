"""
fpi_axes.py — FPI 세 축을 개별 계산하고 어느 축이 우울 판별에 제일 강한지 비교.
축1 전이이상 / 축2 긍정반응둔화(★) / 축3 변동성저하.  전부 '앵커 정렬' 기반.
모델: 상태발견 KMeans(K=6) + 각 축을 로지스틱 5-fold AUC로 평가 (딥러닝 아님, 신호강도 판단용).
"""
import numpy as np, csv
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import mannwhitneyu

D = Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r',
      'AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
I_AU06, I_AU12 = AU.index('AU06_r'), AU.index('AU12_r')  # 미소 근육 (볼+입꼬리)
K = 6; PRE, POST = 3.0, 4.0
POS = ['last time you felt really happy','really happy','proud','enjoy','felt really happy']
NEG = ['diagnosed with depression','feel guilty','regret','made you feel really badly',"don't sleep",'feeling lately']

def labels():
    lab = {}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p = D/f
        if p.exists():
            for r in csv.DictReader(open(p)):
                lab[r['Participant_ID'].strip()] = int(float(r['PHQ8_Binary']))
    return lab

def transcript(pid):
    p = D/f'{pid}_TRANSCRIPT.csv'; rows = []
    if p.exists():
        for r in csv.DictReader(open(p), delimiter='\t'):
            try: rows.append((float(r['start_time']), float(r['stop_time']), r['speaker'].strip(), (r['value'] or '').lower()))
            except: pass
    return rows

def anchors(rows, keys): return [sp for st,sp,spk,val in rows if spk=='Ellie' and any(k in val for k in keys)]

def au_series(pid):
    p = D/f'{pid}_CLNF_AUs.txt'
    if not p.exists(): return None, None
    h = [x.strip() for x in open(p).readline().split(',')]
    ti, oi = h.index('timestamp'), h.index('success'); ai = [h.index(c) for c in AU]
    ts, fe = [], []
    with open(p) as fp:
        fp.readline()
        for ln in fp:
            v = ln.split(',')
            try:
                if int(float(v[oi])) != 1: continue
                ts.append(float(v[ti])); fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(ts), np.array(fe)

# ---- 로드 + 상태 군집 ----
lab = labels(); data = {}; frames = []
for pid in lab:
    ts, au = au_series(pid)
    if ts is None or len(ts) == 0: continue
    data[pid] = (ts, au)
    idx = np.linspace(0, len(au)-1, min(len(au), 800)).astype(int)
    frames.append(au[idx])
frames = np.vstack(frames)
km = KMeans(n_clusters=K, random_state=42, n_init=5).fit(frames)
print(f'참가자 {len(data)}명, 상태 K={K}', flush=True)

def segs_at(pid, keys, pre=0.0, post=POST):
    ts, au = data[pid]; out = []
    for a in anchors(transcript(pid), keys):
        m = (ts >= a-pre) & (ts < a+post)
        if m.sum() >= 4: out.append((au[(ts>=a-pre)&(ts<a)], au[(ts>=a)&(ts<a+post)]))
    return out

def state_dist(au):
    if len(au) == 0: return np.zeros(K)
    s = km.predict(au); return np.bincount(s, minlength=K)/len(s)

def norm_trans(au_seq):
    """상태 전이 행렬 (KxK, row-normalized)."""
    if len(au_seq) < 2: return None
    s = km.predict(au_seq); M = np.zeros((K,K))
    for a,b in zip(s[:-1], s[1:]): M[a,b] += 1
    rs = M.sum(1, keepdims=True); rs[rs==0] = 1
    return M/rs

# ---- 참가자별 3축 계산 (긍정 앵커 기준) ----
X1, X2, X3, y, base_trans_normal = [], [], [], [], []
rows_all = {}
for pid in data:
    pos = segs_at(pid, POS, pre=PRE, post=POST)
    if not pos: continue
    # 축2: 미소(AU06+12) pre->post 증가량 (작을수록 둔화)
    d_smile = []
    post_all = []
    for pre, post in pos:
        if len(pre) >= 2 and len(post) >= 2:
            sm_pre = pre[:, [I_AU06, I_AU12]].mean()
            sm_post = post[:, [I_AU06, I_AU12]].mean()
            d_smile.append(sm_post - sm_pre)
        if len(post) >= 2: post_all.append(post)
    if not d_smile or not post_all: continue
    post_cat = np.vstack(post_all)
    # 축3: 앵커 구간 변동성 (낮을수록 평탄)
    var = post_cat.std(0).mean()
    # 축1: 앵커 구간 전이행렬 (나중에 정상군평균과 거리)
    tr = norm_trans(post_cat)
    if tr is None: continue
    rows_all[pid] = dict(d_smile=np.mean(d_smile), var=var, tr=tr, y=lab[pid])

# 정상군 평균 전이행렬
norm_trs = [v['tr'] for v in rows_all.values() if v['y']==0]
P_norm = np.mean(norm_trs, axis=0)

for pid, v in rows_all.items():
    X2.append(-v['d_smile'])                     # 반응둔화: 미소증가 작을수록 큰 값
    X3.append(-v['var'])                         # 변동성저하: std 작을수록 큰 값
    X1.append(np.abs(v['tr'] - P_norm).sum())    # 전이이상: 정상평균과 거리
    y.append(v['y'])
X1,X2,X3,y = map(np.array, (X1,X2,X3,y))
print(f'3축 계산 완료: n={len(y)} (우울{int(y.sum())}/정상{int((y==0).sum())})\n', flush=True)

def evalax(name, x):
    x = x.reshape(-1,1)
    dep, nor = x[y==1,0], x[y==0,0]
    d = (dep.mean()-nor.mean())/np.sqrt((dep.var(ddof=1)+nor.var(ddof=1))/2)
    try: _,p = mannwhitneyu(dep, nor, alternative='two-sided')
    except: p = np.nan
    auc = cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'), x, y, cv=5, scoring='roc_auc').mean()
    s = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
    print(f'  {name:16s} AUC={auc:.3f}  Cohen_d={d:+.2f}  p={p:.3f}{s}', flush=True)
    return auc

print('=== 축별 단독 판별력 (긍정 앵커 정렬) ===', flush=True)
a1 = evalax('축1 전이이상', X1)
a2 = evalax('축2 반응둔화★', X2)
a3 = evalax('축3 변동성저하', X3)

# FPI 결합 (3축 표준화 후 로지스틱 = 학습된 가중합)
from sklearn.preprocessing import StandardScaler
Xf = StandardScaler().fit_transform(np.c_[X1,X2,X3])
auc_f = cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'), Xf, y, cv=5, scoring='roc_auc').mean()
print(f'\n=== FPI (3축 결합) AUC={auc_f:.3f} ===', flush=True)
print(f'\n[요약] 최강 축 = ' + ['축1전이','축2반응둔화','축3변동성'][np.argmax([a1,a2,a3])], flush=True)
