"""
exp1_anchor.py — 핵심 비교 실험 (교수님 논문식 baseline 비교)
질문: 앵커(긍정질문 정렬)가 실제로 우울 판별을 올리나? + 방법별 비교.
비교축1: 전체인터뷰 vs 긍정앵커 vs 부정앵커
비교축2: 표정상태 분포 / top-p 선택(ExpADA식) / 통계요약
평가: 로지스틱 5-fold AUC, 여러 seed 평균±std (교수님식 통계)
가볍게 = 신호 유무 빠른 확인. 되면 무거운 모델로.
"""
import numpy as np, csv, warnings
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
warnings.filterwarnings('ignore')

D = Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r',
      'AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
K = 8; POST = 4.0
POS = ['last time you felt really happy','really happy','proud','enjoy','felt really happy']
NEG = ['diagnosed with depression','feel guilty','regret','made you feel really badly',"don't sleep",'feeling lately']
SEEDS = [42, 1, 2, 3, 4]

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

# 로드
lab = labels(); data = {}; frames = []
for pid in lab:
    ts, au = au_series(pid)
    if ts is None or len(ts) == 0: continue
    data[pid] = (ts, au)
    idx = np.linspace(0, len(au)-1, min(len(au), 800)).astype(int)
    frames.append(au[idx])
km = KMeans(n_clusters=K, random_state=42, n_init=5).fit(np.vstack(frames))
print(f'참가자 {len(data)}명, 상태 K={K}\n', flush=True)

def get_segment(pid, mode):
    """mode: 'all'(전체) / 'pos'(긍정앵커후) / 'neg'(부정앵커후) → AU 배열"""
    ts, au = data[pid]
    if mode == 'all': return au
    keys = POS if mode == 'pos' else NEG
    segs = []
    for a in anchors(transcript(pid), keys):
        m = (ts >= a) & (ts < a + POST)
        if m.sum() >= 3: segs.append(au[m])
    return np.vstack(segs) if segs else None

def feat_statedist(au):
    s = km.predict(au); return np.bincount(s, minlength=K)/len(s)

def feat_topp(au, p=0.2):
    """ExpADA식: 상태 확신도 높은 상위 p 프레임만 골라 분포 (관련성 선택 흉내)"""
    d = km.transform(au); conf = -d.min(1)  # 가까운 중심까지 거리(작을수록 확신) → 부호반전
    k = max(3, int(len(au)*p)); idx = np.argsort(conf)[-k:]
    s = km.predict(au[idx]); return np.bincount(s, minlength=K)/len(s)

def feat_stat(au):
    """통상 통계 요약: 평균+표준편차"""
    return np.concatenate([au.mean(0), au.std(0)])

def build(mode, featfn):
    X, y = [], []
    for pid, l in lab.items():
        if pid not in data: continue
        seg = get_segment(pid, mode)
        if seg is None or len(seg) < 3: continue
        X.append(featfn(seg)); y.append(l)
    return np.array(X), np.array(y)

def evaluate(X, y):
    aucs = []
    for s in SEEDS:
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        auc = cross_val_score(LogisticRegression(max_iter=1000, class_weight='balanced'),
                              X, y, cv=cv, scoring='roc_auc')
        aucs.append(auc.mean())
    return np.mean(aucs), np.std(aucs), len(y)

print(f"{'앵커':>8} {'특징방법':>14} {'AUC':>14} {'n':>5}", flush=True)
print('-'*48, flush=True)
for mode, mname in [('all','전체인터뷰'), ('pos','긍정앵커'), ('neg','부정앵커')]:
    for featfn, fname in [(feat_statedist,'상태분포'), (feat_topp,'top-p선택'), (feat_stat,'통계요약')]:
        X, y = build(mode, featfn)
        if len(y) < 20: continue
        m, sd, n = evaluate(X, y)
        print(f"{mname:>8} {fname:>14} {m:.3f}±{sd:.3f}   {n:>5}", flush=True)
