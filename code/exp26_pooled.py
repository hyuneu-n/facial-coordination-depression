"""
exp26 — 3데이터 통합 (CMDC + DAIC + E-DAIC) 공분산 우울 판별.
목적: 규모 키우면(500+) 성능/신뢰↑? 다기관·교차언어 일반화 novelty.
공통 AU 14개(세 데이터 교집합)로 공분산 → tangent → 로지스틱.
A) 전체인터뷰 공분산 (3데이터 다, 앵커 불필요)
B) 데이터별 within + pooled 비교
C) leave-one-corpus-out (교차 일반화)
결과 CSV. 예상 5-8분(E-DAIC 로드 느림).
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
LOG=open('/home/hyuneun/disk_b/🟡facial-prodrome/results/pooled_result.csv','w')
def P(*a): print(*a,flush=True)
BASE=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
# 세 데이터 공통 AU 14개 (DAIC CLNF 기준 교집합)
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']

def read_au_csv(path, has_face_id=False):
    h=[x.strip() for x in open(path).readline().split(',')]
    try:
        oi=h.index('success'); ai=[h.index(c) for c in AU]
    except ValueError: return None
    fe=[]
    with open(path) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(fe) if len(fe)>=20 else None

# ---- CMDC ----
def load_cmdc():
    C=BASE/'CMDC/extracted'
    wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
    rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0]); iID,iMDD=hd.index('ID'),hd.index('MDD')
    lab={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
    out=[]
    for subj,l in lab.items():
        segs=[]
        for q in range(1,13):
            f=C/subj/f'Q{q}.csv'
            if f.exists():
                a=read_au_csv(f)
                if a is not None: segs.append(a)
        if segs: out.append((np.vstack(segs),l,'CMDC'))
    return out
# ---- DAIC ----
def load_daic():
    D=BASE/'DAIC_WOZ'; lab={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)): lab[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
    out=[]
    for pid,l in lab.items():
        f=D/f'{pid}_CLNF_AUs.txt'
        if f.exists():
            a=read_au_csv(f)
            if a is not None: out.append((a,l,'DAIC'))
    return out
# ---- E-DAIC ----
def load_edaic():
    E=BASE/'E-DAIC'; lab={}
    for f in ['train_split.csv','dev_split.csv','test_split.csv']:
        p=E/'labels'/f
        if p.exists():
            for r in csv.DictReader(open(p)):
                pid=r['Participant_ID'].strip(); b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
                if b not in (None,''): lab[pid]=int(float(b))
    out=[]
    for pid,l in lab.items():
        fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
        if fs:
            a=read_au_csv(fs[0])
            if a is not None: out.append((a,l,'E-DAIC'))
    return out

P('로딩...(E-DAIC 느림)')
data=load_cmdc()+load_daic()+load_edaic()
from collections import Counter
srcs=Counter(d[2] for d in data)
P(f'총 {len(data)}명: {dict(srcs)}, 우울 {sum(d[1] for d in data)}')

# 공분산 → 피처
covs=[]; ys=[]; src=[]
for X,l,s in data:
    Xn=(X-X.mean(0))/(X.std(0)+1e-6)
    c,_=ledoit_wolf(Xn); covs.append(c); ys.append(l); src.append(s)
covs=np.array(covs); ys=np.array(ys); src=np.array(src)

def auc_cv(cv,yy,seeds=range(5)):
    aucs=[]
    for sd in seeds:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(yy))
        for tr,te in skf.split(cv,yy):
            ts=TangentSpace(metric='riemann').fit(cv[tr])
            Xtr=ts.transform(cv[tr]); Xte=ts.transform(cv[te])
            sc=StandardScaler().fit(Xtr)
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),yy[tr])
            pb[te]=clf.decision_function(sc.transform(Xte))
        aucs.append(roc_auc_score(yy,pb))
    return np.mean(aucs),np.std(aucs)

LOG.write('setting,AUC,std,n\n')
P('\n=== 전체인터뷰 공분산 (앵커 없이) ===')
# 데이터별 within
for s in ['CMDC','DAIC','E-DAIC']:
    m=src==s
    if m.sum()>20:
        a,sd=auc_cv(covs[m],ys[m]); P(f'  {s} within: AUC={a:.3f}±{sd:.3f} (n={m.sum()})'); LOG.write(f'{s}_within,{a:.4f},{sd:.4f},{m.sum()}\n')
# pooled 전체
a,sd=auc_cv(covs,ys); P(f'  POOLED 전체: AUC={a:.3f}±{sd:.3f} (n={len(ys)})'); LOG.write(f'pooled_all,{a:.4f},{sd:.4f},{len(ys)}\n')

# leave-one-corpus-out (교차 일반화)
P('\n=== Leave-One-Corpus-Out (교차 일반화) ===')
for test_s in ['CMDC','DAIC','E-DAIC']:
    trm=src!=test_s; tem=src==test_s
    if tem.sum()<20: continue
    ts=TangentSpace(metric='riemann').fit(covs[trm]); Xtr=ts.transform(covs[trm]); Xte=ts.transform(covs[tem])
    sc=StandardScaler().fit(Xtr)
    clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),ys[trm])
    a=roc_auc_score(ys[tem],clf.decision_function(sc.transform(Xte)))
    P(f'  train(2corpus)→test {test_s}: AUC={a:.3f} (n_test={tem.sum()})'); LOG.write(f'LOCO_{test_s},{a:.4f},0,{tem.sum()}\n')
LOG.close(); P('\nDONE (CSV saved)')
