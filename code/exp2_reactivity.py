"""
exp2_reactivity.py — 핵심 아이디어 직접 구현: '반응성'(pre->post 변화) + 개인정규화
지금까지 안 쓴 우리 진짜 가설: 긍정질문 '전 vs 후' 표정이 얼마나 변하나(=반응).
이게 0.58 넘겨 오르면 방향 맞음, 아니면 표정단독 한계.
"""
import numpy as np, csv, warnings
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
warnings.filterwarnings('ignore')

D = Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r',
      'AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
PRE, POST = 4.0, 5.0
POS = ['last time you felt really happy','really happy','proud','enjoy','felt really happy']
SEEDS = [42,1,2,3,4]

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

lab=labels(); data={}
for pid in lab:
    ts,au=au_series(pid)
    if ts is not None and len(ts)>0: data[pid]=(ts,au)
print(f'참가자 {len(data)}명\n',flush=True)

def reactivity_feat(pid, personal_norm=True):
    ts,au=data[pid]
    base_mean=au.mean(0); base_std=au.std(0)+1e-6  # 개인 baseline
    reacts=[]
    for a in anchors(transcript(pid),POS):
        pre=au[(ts>=a-PRE)&(ts<a)]; post=au[(ts>=a)&(ts<a+POST)]
        if len(pre)<2 or len(post)<2: continue
        if personal_norm:
            pre=(pre-base_mean)/base_std; post=(post-base_mean)/base_std
        # 반응 특징: post평균-pre평균(변화), post변동성, post움직임
        delta=post.mean(0)-pre.mean(0)
        postvar=post.std(0)
        motion=np.abs(np.diff(post,axis=0)).mean(0) if len(post)>1 else np.zeros(len(AU))
        reacts.append(np.concatenate([delta,postvar,motion]))
    if not reacts: return None
    return np.mean(reacts,axis=0)

def evaluate(X,y):
    aucs=[]
    for s in SEEDS:
        cv=StratifiedKFold(5,shuffle=True,random_state=s)
        aucs.append(cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'),X,y,cv=cv,scoring='roc_auc').mean())
    return np.mean(aucs),np.std(aucs),len(y)

for pn,label in [(False,'정규화X'),(True,'개인정규화O')]:
    X,y=[],[]
    for pid,l in lab.items():
        if pid not in data: continue
        f=reactivity_feat(pid,pn)
        if f is not None: X.append(f); y.append(l)
    X,y=np.array(X),np.array(y)
    m,sd,n=evaluate(X,y)
    print(f'[반응성(pre->post변화) {label}] AUC={m:.3f}±{sd:.3f}  n={n}',flush=True)
