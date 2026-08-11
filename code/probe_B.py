"""
probe_B.py — 가정 B 검증: 앵커(Ellie 부정정서 질문) 직후 참가자 응답 구간의
AU가 우울군 vs 정상군에서 다른가? (모델 없이 통계·효과크기만)

앵커: Ellie 발화 중 아래 질문 태그/문구가 나온 시점.
그 직후 '참가자'가 답하는 구간(다음 Participant 발화들, 최대 W초)의 AU 평균/표준편차를
우울(PHQ8_Binary=1) vs 정상(0) 비교. Mann-Whitney U + Cohen's d.
근거: DAIC transcript는 Ellie 질문이 정형화됨(happy_lasttime, depression_diagnosed 등).
"""
import numpy as np, csv, glob, re
from pathlib import Path
from scipy.stats import mannwhitneyu

D = Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
WINDOW = 8.0  # 앵커 직후 몇 초를 응답 구간으로 볼지

# 부정 정서 유도 앵커 (Ellie 질문 문구 부분일치, 소문자)
NEG_ANCHORS = [
    'diagnosed with depression', 'feel guilty', 'something you feel guilty',
    'regret', 'made you feel really badly', 'when you don', "don't sleep",
    'how have you been feeling', 'feel in that moment',
]
AU_COLS = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r',
           'AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']  # 강도(_r)만

def load_labels():
    lab = {}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p = D/f
        if not p.exists(): continue
        with open(p) as fp:
            for r in csv.DictReader(fp):
                pid = r['Participant_ID'].strip()
                lab[pid] = int(float(r['PHQ8_Binary']))
    return lab

def load_transcript(pid):
    p = D/f'{pid}_TRANSCRIPT.csv'
    rows = []
    if not p.exists(): return rows
    with open(p) as fp:
        rd = csv.DictReader(fp, delimiter='\t')
        for r in rd:
            try:
                rows.append((float(r['start_time']), float(r['stop_time']),
                             r['speaker'].strip(), (r['value'] or '').strip().lower()))
            except: pass
    return rows

def find_anchors(rows):
    """Ellie가 부정정서 질문을 한 시점(stop_time)들 반환 = 참가자 응답 시작 기준."""
    anchors = []
    for st, sp, spk, val in rows:
        if spk == 'Ellie' and any(a in val for a in NEG_ANCHORS):
            anchors.append(sp)  # 질문 끝난 시점부터 참가자 답변
    return anchors

def load_au(pid):
    p = D/f'{pid}_CLNF_AUs.txt'
    if not p.exists(): return None, None
    with open(p) as fp:
        header = [h.strip() for h in fp.readline().split(',')]
        idx_t = header.index('timestamp')
        idx_ok = header.index('success')
        au_idx = [header.index(c) for c in AU_COLS]
        ts, feats = [], []
        for line in fp:
            v = line.split(',')
            try:
                if int(float(v[idx_ok])) != 1: continue
                ts.append(float(v[idx_t]))
                feats.append([float(v[i]) for i in au_idx])
            except: pass
    return np.array(ts), np.array(feats)

def anchor_response_feats(pid, anchors):
    ts, au = load_au(pid)
    if ts is None or len(ts) == 0 or not anchors: return None
    segs = []
    for a in anchors:
        m = (ts >= a) & (ts < a + WINDOW)
        if m.sum() >= 5:
            segs.append(au[m])
    if not segs: return None
    allf = np.vstack(segs)
    # 참가자별 요약: 각 AU의 평균 + 표준편차(변동성)
    return np.concatenate([allf.mean(0), allf.std(0)])

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    sp = np.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2))
    return (a.mean()-b.mean())/sp if sp > 0 else np.nan

def main():
    lab = load_labels()
    print(f'라벨 보유: {len(lab)}명 (우울 {sum(lab.values())}, 정상 {len(lab)-sum(lab.values())})', flush=True)
    X, y, used = [], [], 0
    for pid in lab:
        rows = load_transcript(pid)
        anchors = find_anchors(rows)
        f = anchor_response_feats(pid, anchors)
        if f is not None:
            X.append(f); y.append(lab[pid]); used += 1
    X = np.array(X); y = np.array(y)
    print(f'앵커+AU 확보: {used}명 (우울 {int(y.sum())}, 정상 {int((y==0).sum())})', flush=True)
    names = [f'{c}_mean' for c in AU_COLS] + [f'{c}_std' for c in AU_COLS]
    dep, nor = X[y==1], X[y==0]
    print('\n=== 우울 vs 정상: 앵커 직후 응답 AU 차이 (|d|>=0.3만, 큰 순) ===', flush=True)
    res = []
    for i, nm in enumerate(names):
        d = cohens_d(dep[:,i], nor[:,i])
        try: _, p = mannwhitneyu(dep[:,i], nor[:,i], alternative='two-sided')
        except: p = np.nan
        res.append((nm, d, p, dep[:,i].mean(), nor[:,i].mean()))
    res.sort(key=lambda r: -abs(r[1]) if not np.isnan(r[1]) else 0)
    for nm, d, p, md, mn in res:
        if not np.isnan(d) and abs(d) >= 0.3:
            star = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
            print(f'  {nm:14s} d={d:+.2f} p={p:.3f}{star}  우울={md:.3f} 정상={mn:.3f}', flush=True)
    sig = [r for r in res if not np.isnan(r[2]) and r[2]<0.05]
    print(f'\n유의미(p<0.05) 특징: {len(sig)}/{len(names)}개', flush=True)
    np.save('/home/hyuneun/disk_b/🟡facial-prodrome/features/probeB_X.npy', X)
    np.save('/home/hyuneun/disk_b/🟡facial-prodrome/features/probeB_y.npy', y)
    print('저장: features/probeB_X.npy, probeB_y.npy', flush=True)

if __name__ == '__main__':
    main()
