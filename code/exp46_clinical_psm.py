"""
exp46 — 임상(CMDC/DAIC/E-DAIC)서 정신운동 precursor 재현.
AU 시계열 속도로 psychomotor 특징(exp45와 동일 8개) → 기전 방향(MDD<HC?) + AUC.
핵심 확인: pre_ratio(lead-up이 baseline보다 느림)가 임상서도 우울에서 낮은가(=robust).
결과: results/exp46_clinical_psm.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); W=15; GAP=2
FN=['spd_mean','spd_std','spd_med','event_rate','event_amp','pre_spd','pre_ratio','decel']
def psm(AU):  # AU:(T,d) 한 subject 전체
    if len(AU)<40: return None
    Z=(AU-AU.mean(0))/(AU.std(0)+1e-6)
    spd=np.linalg.norm(np.diff(Z,axis=0),axis=1); g=spd.mean()+1e-6
    thr=np.percentile(spd,80); ev=[]; t=W+GAP
    while t<len(spd)-1:
        if spd[t]>=thr and spd[t]>=spd[t-1] and spd[t]>=spd[t+1]: ev.append(t);t+=W
        else:t+=1
    rate=len(ev)/len(spd); amp=np.mean([spd[e] for e in ev]) if ev else g
    pre=[spd[e-W-GAP:e-GAP].mean() for e in ev if e-W-GAP>=0]; prem=np.mean(pre) if pre else g
    decel=np.mean([spd[e]-spd[e-W-GAP:e-GAP].mean() for e in ev if e-W-GAP>=0]) if pre else 0
    return np.array([spd.mean(),spd.std(),np.median(spd),rate,amp,prem,prem/g,decel])
def run(subjAU, y, name):
    F=[];yy=[]
    for i,au in enumerate(subjAU):
        f=psm(au)
        if f is not None: F.append(f);yy.append(y[i])
    F=np.nan_to_num(np.array(F));yy=np.array(yy)
    print(f'\n=== {name} n={len(yy)}(dep{yy.sum()}) ===',flush=True)
    for j,nm in enumerate(FN):
        md,hc=F[yy==1,j].mean(),F[yy==0,j].mean(); mark='↓' if md<hc else '↑'
        star=' ★' if nm in('spd_mean','pre_ratio') else ''
        print(f'  {nm:11s} MDD={md:.3f} HC={hc:.3f} ({mark}우울){star}',flush=True)
    a=[]
    for s in range(10):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(yy))
        for tr,te in skf.split(F,yy):
            sc=StandardScaler().fit(F[tr]);cl=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(F[tr]),yy[tr])
            pb[te]=cl.decision_function(sc.transform(F[te]))
        a.append(roc_auc_score(yy,pb))
    print(f'  정신운동 AUC={np.mean(a):.3f}±{np.std(a):.3f}',flush=True)
    return name,len(yy),np.mean(a),F[yy==1,6].mean(),F[yy==0,6].mean()

res=[]
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
# CMDC (전체 Q concat)
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
sA=[];sy=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)]; parts=[p for p in parts if p is not None]
    if parts: sA.append(np.vstack(parts));sy.append(l)
res.append(run(sA,sy,'CMDC'))
# DAIC (whole success)
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
def dau(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None
    h=[x.strip() for x in open(p).readline().split(',')];oi=h.index('success');ai=[h.index(c) for c in AUd]
    fe=[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if fe else None
sA=[];sy=[]
for pid,l in dl.items():
    au=dau(pid)
    if au is not None: sA.append(au);sy.append(l)
res.append(run(sA,sy,'DAIC'))
# E-DAIC (whole, quality gate)
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
def eau(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
    if not fs:return None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[]
    for ln in open(fs[0]).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if fe else None
sA=[];sy=[]
for pid,l in el.items():
    au=eau(pid)
    if au is not None: sA.append(au);sy.append(l)
res.append(run(sA,sy,'E-DAIC'))
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp46_clinical_psm.csv','w') as f:
    f.write('dataset,n,psm_AUC,pre_ratio_MDD,pre_ratio_HC\n')
    for nm,n,a,rm,rh in res:f.write(f'{nm},{n},{a:.4f},{rm:.4f},{rh:.4f}\n')
print('\nDONE → exp46_clinical_psm.csv',flush=True)
