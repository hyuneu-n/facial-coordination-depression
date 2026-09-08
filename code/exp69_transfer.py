"""
exp69 — 강화: expressivity가 coupling과 달리 코퍼스 넘어 전이되나?
coupling은 전이 실패(exp56). 간단한 활동마커(tension=평균AU)가 전이되면 = 일반화 강점.
공통 14 AU. tension=subject 평균AU를 코퍼스별 z-표준화(도메인 정렬, 무라벨). src학습→tgt 제로샷.
비교: expressivity 전이 AUC vs coupling(tangent) 전이 AUC. 4 AU 코퍼스.
결과: results/exp69_transfer.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
COM=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
def cov_of(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def within(F,y):  # within-corpus CV AUC (ref)
    F=np.nan_to_num(F);y=np.array(y);a=[]
    for s in range(5):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            sc=StandardScaler().fit(F[tr]);cl=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(F[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(F[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a)
D={}  # corpus -> (tension[n,14] corpus-z, COV[n], y)
def register(name, AUs, y):
    ten=np.array([au.mean(0) for au in AUs]); ten=(ten-ten.mean(0))/(ten.std(0)+1e-6)  # 코퍼스 z
    COV=np.array([cov_of(au) for au in AUs]); D[name]=(np.nan_to_num(ten),COV,np.array(y))
    print(f'{name}: n={len(y)} 우울{sum(y)}',flush=True)
# ---- load 4 AU corpora (공통 14) ----
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in COM]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
A=[];Y=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: A.append(np.vstack(parts));Y.append(l)
register('CMDC',A,Y)
Dd=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=Dd/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
def dau(pid):
    p=Dd/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None
    h=[x.strip() for x in open(p).readline().split(',')];oi=h.index('success')
    try:ai=[h.index(c) for c in COM]
    except:return None
    fe=[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if fe else None
A=[];Y=[]
for pid,l in dl.items():
    au=dau(pid)
    if au is not None and len(au)>=60: A.append(au);Y.append(l)
register('DAIC',A,Y)
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
def eau(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*.csv'))
    if not fs:return None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in COM]
    except:return None
    fe=[]
    for ln in open(fs[0]).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if len(fe)>=60 else None
A=[];Y=[]
for pid,l in el.items():
    au=eau(pid)
    if au is not None: A.append(au);Y.append(l)
register('E-DAIC',A,Y)
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423): return 1
    if (602<=i<=1116) or (1425<=i<=1824): return 0
    return None
A=[];Y=[]
for f in sorted(glob.glob(str(V/'*.csv'))):
    l=lab(int(Path(f).stem))
    if l is None:continue
    h=[x.strip() for x in open(f).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in COM]
    except:continue
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    if len(fe)>=60: A.append(np.nan_to_num(np.array(fe)));Y.append(l)
register('LMVD',A,Y)
corp=list(D)
print('\n=== 전이 AUC (src→tgt): expressivity | coupling ===',flush=True)
out=[]
for src in corp:
    ts,Cs,ys=D[src]
    for tgt in corp:
        if src==tgt:continue
        tt,Ct,yt=D[tgt]
        # expressivity 전이
        sc=StandardScaler().fit(ts);cle=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(ts),ys)
        ae=roc_auc_score(yt,cle.decision_function(sc.transform(tt)))
        # coupling 전이(tangent src fit)
        T=TangentSpace(metric='riemann').fit(Cs);Xs=T.transform(Cs);Xt=T.transform(Ct)
        sc2=StandardScaler().fit(Xs);clc=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc2.transform(Xs),ys)
        ac=roc_auc_score(yt,clc.decision_function(sc2.transform(Xt)))
        mark='★expr>coupling' if ae>ac+0.03 else ''
        print(f'  {src:6s}→{tgt:6s}  expr={ae:.3f} | coupling={ac:.3f}  {mark}',flush=True)
        out.append([src,tgt,ae,ac])
# 요약
ae_all=np.mean([r[2] for r in out]);ac_all=np.mean([r[3] for r in out])
print(f'\n평균 전이: expressivity={ae_all:.3f} | coupling={ac_all:.3f}  Δ{ae_all-ac_all:+.3f}',flush=True)
print(f'expr>coupling 인 쌍: {sum(1 for r in out if r[2]>r[3]+0.02)}/{len(out)}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp69_transfer.csv','w') as f:
    f.write('src,tgt,expr_auc,coupling_auc\n')
    for r in out:f.write(f'{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f}\n')
print('DONE → exp69_transfer.csv',flush=True)
