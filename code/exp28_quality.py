"""
exp28 — E-DAIC 품질 진단 (조사 1순위). E-DAIC 0.5가 품질 문제인가?
1) 참가자별 coverage(=success·high-conf·frontal 비율) 계산
2) coverage 높은 순으로 자르며 AUC 변화 (AUC-vs-threshold 곡선)
3) frontal+high-conf 프레임만으로 공분산 재계산 → AUC (전체 0.5 대비)
품질 게이트가 E-DAIC 살리나. 결과 CSV.
"""
import numpy as np, warnings, csv, glob
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
E=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/E-DAIC')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
CONF_THR=0.90; POSE_THR=20*np.pi/180  # frontal: |R|<20deg (rad)
SEEDS=list(range(10))
def labels():
    lab={}
    for f in ['train_split.csv','dev_split.csv','test_split.csv']:
        p=E/'labels'/f
        if p.exists():
            for r in csv.DictReader(open(p)):
                pid=r['Participant_ID'].strip(); b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
                if b not in (None,''): lab[pid]=int(float(b))
    return lab
def load(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
    if not fs: return None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try:
        ci=h.index('confidence'); oi=h.index('success'); ai=[h.index(c) for c in AU]
        rx,ry,rz=h.index('pose_Rx'),h.index('pose_Ry'),h.index('pose_Rz')
    except: return None
    au=[];conf=[];ok=[];pose=[]
    with open(fs[0]) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                au.append([float(v[i]) for i in ai]); conf.append(float(v[ci])); ok.append(int(float(v[oi])))
                pose.append([abs(float(v[rx])),abs(float(v[ry])),abs(float(v[rz]))])
            except: pass
    if len(au)<30: return None
    return np.array(au),np.array(conf),np.array(ok),np.array(pose)
lab=labels()
DATA={}
for pid in lab:
    d=load(pid)
    if d is not None: DATA[pid]=d
print(f'E-DAIC 로드: {len(DATA)}명',flush=True)
# coverage 계산
cov_idx={}
for pid,(au,conf,ok,pose) in DATA.items():
    good=(ok==1)&(conf>CONF_THR)&(pose<POSE_THR).all(1)
    cov_idx[pid]=good.mean()
covs_arr=np.array([cov_idx[p] for p in DATA])
print(f'coverage 분포: min={covs_arr.min():.2f} median={np.median(covs_arr):.2f} max={covs_arr.max():.2f}',flush=True)

def make_cov(pid,frontal=False):
    au,conf,ok,pose=DATA[pid]
    if frontal:
        m=(ok==1)&(conf>CONF_THR)&(pose<POSE_THR).all(1)
    else:
        m=ok==1
    if m.sum()<20: return None
    seg=au[m]; seg=(seg-seg.mean(0))/(seg.std(0)+1e-6)
    c,_=ledoit_wolf(seg); return c

def auc_of(pids,frontal):
    C=[];y=[]
    for pid in pids:
        c=make_cov(pid,frontal)
        if c is not None: C.append(c); y.append(lab[pid])
    if len(set(y))<2 or len(y)<25: return None,len(y)
    C=np.array(C);y=np.array(y); aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            ts_=TangentSpace(metric='riemann').fit(C[tr]); Xtr=ts_.transform(C[tr]);Xte=ts_.transform(C[te])
            sc=StandardScaler().fit(Xtr)
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),y[tr])
            pb[te]=clf.decision_function(sc.transform(Xte))
        aucs.append(roc_auc_score(y,pb))
    return np.mean(aucs),len(y)

allpids=list(DATA)
print('\n=== 1) coverage threshold별 AUC (frontal 공분산) ===',flush=True)
res=[]
for thr in [0.0,0.3,0.5,0.6,0.7]:
    keep=[p for p in allpids if cov_idx[p]>=thr]
    a,n=auc_of(keep,frontal=True)
    if a is not None: print(f'  coverage>={thr}: n={n} AUC={a:.3f}',flush=True); res.append((thr,n,a))
print('\n=== 2) 필터 비교 (전체 참가자) ===',flush=True)
a0,n0=auc_of(allpids,frontal=False); print(f'  success만(기존): n={n0} AUC={a0:.3f}',flush=True)
a1,n1=auc_of(allpids,frontal=True); print(f'  frontal+highconf: n={n1} AUC={a1:.3f}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/edaic_quality.csv','w') as f:
    f.write('setting,n,AUC\n')
    f.write(f'success_all,{n0},{a0:.4f}\n'); f.write(f'frontal_highconf,{n1},{a1:.4f}\n')
    for thr,n,a in res: f.write(f'coverage>={thr},{n},{a:.4f}\n')
print('DONE',flush=True)
