"""
probe_C — 표정 '상태' 클러스터링 + 개인정규화로 우울/정상이 갈리나
1) 전체 참가자 AU 프레임을 K개 표정상태로 비지도 군집(KMeans)
2) 참가자별 '상태 체류 분포'(각 상태에 머문 비율) 계산
3) 개인 baseline(전체 평균) 제거 후, 부정/긍정 앵커 응답 구간의 상태분포를 우울 vs 정상 비교
4) 상태분포로 우울 예측 가능한지 (로지스틱 5-fold AUC) — '발견→예측' 가능성 확인
모델은 KMeans/LogReg 수준(딥러닝 아님). 실제 신호 유무 판단용.
"""
import numpy as np, csv
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import mannwhitneyu

D = Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r',
    'AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
K=6; POST=4.0
NEG=['diagnosed with depression','feel guilty','regret','made you feel really badly',"don't sleep",'feeling lately']
POS=['last time you felt really happy','really happy','proud','enjoy']

def labels():
    lab={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)): lab[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
    return lab

def transcript(pid):
    p=D/f'{pid}_TRANSCRIPT.csv'; rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try: rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except: pass
    return rows

def anchors(rows,keys): return [sp for st,sp,spk,val in rows if spk=='Ellie' and any(k in val for k in keys)]

def au_series(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists(): return None,None
    h=[x.strip() for x in open(p).readline().split(',')]
    ti,oi=h.index('timestamp'),h.index('success'); ai=[h.index(c) for c in AU]
    ts,fe=[],[]
    with open(p) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                ts.append(float(v[ti])); fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(ts),np.array(fe)

lab=labels()
# --- 1) 전체 프레임 모아 상태 군집 (참가자별 다운샘플로 메모리 관리) ---
data={}
allframes=[]
for pid in lab:
    ts,au=au_series(pid)
    if ts is None or len(ts)==0: continue
    data[pid]=(ts,au)
    idx=np.linspace(0,len(au)-1,min(len(au),800)).astype(int)
    allframes.append(au[idx])
allframes=np.vstack(allframes)
km=KMeans(n_clusters=K,random_state=42,n_init=5).fit(allframes)
print(f'상태 군집 K={K}, 참가자 {len(data)}명, 군집표본 {len(allframes)}',flush=True)

def state_dist(au):
    if len(au)==0: return None
    s=km.predict(au); return np.bincount(s,minlength=K)/len(s)

def anchor_state_dist(pid,keys):
    ts,au=data[pid]; anch=anchors(transcript(pid),keys)
    segs=[au[(ts>=a)&(ts<a+POST)] for a in anch]
    segs=[s for s in segs if len(s)>=4]
    if not segs: return None
    d=state_dist(np.vstack(segs))
    base=state_dist(au)  # 개인 baseline
    return d-base  # 개인정규화된 상태분포 편차

def compare(name,keys):
    X,y=[],[]
    for pid,l in lab.items():
        if pid not in data: continue
        d=anchor_state_dist(pid,keys)
        if d is not None: X.append(d); y.append(l)
    X=np.array(X); y=np.array(y); dep,nor=X[y==1],X[y==0]
    print(f'\n=== [{name}] n={len(y)} (우울{int(y.sum())}/정상{int((y==0).sum())}) 상태체류 편차 우울vs정상 ===',flush=True)
    best=[]
    for k in range(K):
        try: _,p=mannwhitneyu(dep[:,k],nor[:,k],alternative='two-sided')
        except: p=1
        diff=dep[:,k].mean()-nor[:,k].mean()
        best.append((k,diff,p))
    for k,diff,p in sorted(best,key=lambda r:r[2]):
        s='***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        print(f'  상태{k}: 우울-정상={diff:+.3f} p={p:.3f}{s}',flush=True)
    # 예측 가능성
    if len(np.unique(y))==2:
        auc=cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'),X,y,cv=5,scoring='roc_auc')
        print(f'  → 상태분포로 우울 예측 AUC = {auc.mean():.3f} ± {auc.std():.3f}',flush=True)

compare('부정정서 앵커',NEG)
compare('긍정(happy) 앵커',POS)
# 전체 인터뷰(앵커 무관) 상태분포로도 예측 되나 (베이스라인)
Xa,ya=[],[]
for pid,l in lab.items():
    if pid not in data: continue
    d=state_dist(data[pid][1])
    if d is not None: Xa.append(d); ya.append(l)
Xa,ya=np.array(Xa),np.array(ya)
auc=cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'),Xa,ya,cv=5,scoring='roc_auc')
print(f'\n[전체 인터뷰 상태분포] 우울 예측 AUC = {auc.mean():.3f} ± {auc.std():.3f} (앵커 무관 베이스라인)',flush=True)
