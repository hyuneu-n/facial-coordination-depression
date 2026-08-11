"""
exp6 — 선형대수 지수 (신경망 X). 표정 '저차원성/단조로움'을 우울 지표로.
가설: 우울군은 표정 변화가 저차원(단조) → AU 행렬 특이값 분포로 정량화.
지수들:
 - eff_rank: 유효 랭크 (특이값 엔트로피 exp) — 낮을수록 단조
 - sv_entropy: 특이값 분포 엔트로피
 - top1_ratio: 첫 특이값 비중 (높을수록 한 방향 지배=단조)
 - participation_ratio: (Σσ²)²/Σσ⁴
각 지수 단독 AUC + 결합. 긍정앵커. 손절 아님, 1차 시도.
"""
import numpy as np, csv, warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
D=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
POST=6.0; SEEDS=[42,1,2,3,4]
POS=['last time you felt really happy','really happy','proud','enjoy','felt really happy']

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

def linalg_feats(seg):
    # seg: (T, AU). 시간축 변화의 특이값 분석 (평균제거 후)
    X=seg-seg.mean(0,keepdims=True)
    if X.shape[0]<3: return None
    s=np.linalg.svd(X,compute_uv=False)
    s=s[s>1e-8]
    if len(s)<2: return None
    p=s/s.sum()
    sv_entropy=-(p*np.log(p+1e-12)).sum()
    eff_rank=np.exp(sv_entropy)
    top1=p[0]
    s2=s**2; part_ratio=(s2.sum()**2)/((s2**2).sum())
    return np.array([eff_rank, sv_entropy, top1, part_ratio])

lab=labels(); X,y=[],[]
for pid,l in lab.items():
    ts,au=au_series(pid)
    if ts is None: continue
    base=au.mean(0); bs=au.std(0)+1e-6
    segs=[au[(ts>=a)&(ts<a+POST)] for a in anchors(transcript(pid),POS)]
    segs=[s for s in segs if len(s)>=4]
    if not segs: continue
    seg=(np.vstack(segs)-base)/bs
    f=linalg_feats(seg)
    if f is not None: X.append(f); y.append(l)
X,y=np.array(X),np.array(y)
names=['유효랭크','특이값엔트로피','top1비중','participation']
print(f'n={len(y)}, 우울{int(y.sum())}/정상{int((y==0).sum())}\n',flush=True)

# 각 지수 단독 방향+판별
print('=== 지수별 우울vs정상 (단독) ===',flush=True)
for i,nm in enumerate(names):
    dep,nor=X[y==1,i],X[y==0,i]
    try: _,p=mannwhitneyu(dep,nor,alternative='two-sided')
    except: p=1
    aucs=[]
    for s in SEEDS:
        cv=StratifiedKFold(5,shuffle=True,random_state=s)
        aucs.append(cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'),X[:,[i]],y,cv=cv,scoring='roc_auc').mean())
    star='**' if p<0.05 else '*' if p<0.1 else ''
    print(f'  {nm:14s} AUC={np.mean(aucs):.3f}  우울={dep.mean():.3f} 정상={nor.mean():.3f} p={p:.3f}{star}',flush=True)

# 결합
aucs=[]
for s in SEEDS:
    cv=StratifiedKFold(5,shuffle=True,random_state=s)
    aucs.append(cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'),X,y,cv=cv,scoring='roc_auc').mean())
print(f'\n[선형대수 지수 결합] AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)
print(f'(비교: ST-GCN 0.61, AU+gaze+pose 0.63)',flush=True)
