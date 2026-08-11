"""
exp4_edaic.py — E-DAIC(275명, OpenFace2.0) 규모/품질 효과 확인
앵커 아직 없음(E-DAIC transcript에 질문자 없음) → 전체 인터뷰 AU로 먼저.
DAIC 전체(0.40) 대비 규모·품질로 오르나? 오르면 앵커 투자 가치.
방법: 상태분포 + 통계요약, ST-GCN. 5-fold 여러 seed.
"""
import numpy as np, csv, glob, warnings
from pathlib import Path
import torch, torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

E=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/E-DAIC')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r',
    'AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
SEEDS=[42,1,2]; K=8; T=64

def labels():
    lab={}
    for f in ['train_split.csv','dev_split.csv','test_split.csv']:
        p=E/'labels'/f
        if p.exists():
            for r in csv.DictReader(open(p)):
                pid=r['Participant_ID'].strip()
                b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
                if b not in (None,''): lab[pid]=int(float(b))
    return lab

def au_series(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*Pose_gaze_AUs.csv'))
    if not fs: return None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try: ti,oi=h.index('timestamp'),h.index('success'); ai=[h.index(c) for c in AU]
    except: return None
    fe=[]
    with open(fs[0]) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(fe) if fe else None

lab=labels(); data={}
for pid in lab:
    au=au_series(pid)
    if au is not None and len(au)>10: data[pid]=au
print(f'E-DAIC 로드: {len(data)}명 (우울 {sum(lab[p] for p in data)}/정상 {sum(1-lab[p] for p in data)})',flush=True)
if len(data)<30:
    print('아직 압축 덜 풀림 or 로드 부족 — 나중에 재실행',flush=True); raise SystemExit

frames=[]
for pid,au in data.items():
    idx=np.linspace(0,len(au)-1,min(len(au),800)).astype(int); frames.append(au[idx])
km=KMeans(K,random_state=42,n_init=5).fit(np.vstack(frames))
NA=len(AU)

def statedist(au): s=km.predict(au); return np.bincount(s,minlength=K)/len(s)
def stat(au): return np.concatenate([au.mean(0),au.std(0)])

def ev(featfn,name):
    X=[featfn(data[p]) for p in data]; y=[lab[p] for p in data]
    X,y=np.array(X),np.array(y); aucs=[]
    for s in SEEDS:
        cv=StratifiedKFold(5,shuffle=True,random_state=s)
        aucs.append(cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'),X,y,cv=cv,scoring='roc_auc').mean())
    print(f'  [{name}] AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)

print('=== E-DAIC 전체인터뷰 (앵커無) ===',flush=True)
ev(statedist,'상태분포'); ev(stat,'통계요약')
