"""
exp27 — E-DAIC에 '증상 언급 앵커' 만들기 (CMDC와 통합 열쇠).
참가자 답변 transcript에서 sleep/tired/energy/appetite/interest 언급 발화 시점 →
그 구간 AU 공분산 → 우울 판별. 전체인터뷰(0.495) 대비 오르나?
되면 CMDC(수면·피로 앵커)와 같은 증상축 = 통합 가능.
결과 CSV. 예상 3-5분(E-DAIC 로드).
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
SYMPTOM=['sleep','tired','fatigue','energy','appetite','eating','interest','concentrat']  # 증상축 단어
WIN=7.0; SEEDS=list(range(10))
def labels():
    lab={}
    for f in ['train_split.csv','dev_split.csv','test_split.csv']:
        p=E/'labels'/f
        if p.exists():
            for r in csv.DictReader(open(p)):
                pid=r['Participant_ID'].strip(); b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
                if b not in (None,''): lab[pid]=int(float(b))
    return lab
def transcript(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/f'{pid}_Transcript.csv'))
    if not fs: return []
    rows=[]
    for r in csv.DictReader(open(fs[0])):
        try: rows.append((float(r['Start_Time']),float(r['End_Time']),(r['Text'] or '').lower()))
        except: pass
    return rows
def au_full(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
    if not fs: return None,None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try: ti=h.index('timestamp'); oi=h.index('success'); ai=[h.index(c) for c in AU]
    except: return None,None
    ts,fe=[],[]
    with open(fs[0]) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                ts.append(float(v[ti])); fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(ts),np.array(fe)
lab=labels()
def cov_of(pid,mode):
    ts,au=au_full(pid)
    if ts is None or len(au)<30: return None
    if mode=='all': seg=au
    else:
        rows=transcript(pid); segs=[]
        for st,en,txt in rows:
            if any(w in txt for w in SYMPTOM):
                m=(ts>=st)&(ts<en+WIN)  # 증상 언급 발화 + 직후
                if m.sum()>=8: segs.append(au[m])
        if not segs: return None
        seg=np.vstack(segs)
    if len(seg)<20: return None
    seg=(seg-seg.mean(0))/(seg.std(0)+1e-6)
    c,_=ledoit_wolf(seg); return c
def run(mode,name):
    C=[];y=[]
    for pid,l in lab.items():
        c=cov_of(pid,mode)
        if c is not None: C.append(c); y.append(l)
    C=np.array(C); y=np.array(y)
    aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            ts_=TangentSpace(metric='riemann').fit(C[tr])
            Xtr=ts_.transform(C[tr]); Xte=ts_.transform(C[te]); sc=StandardScaler().fit(Xtr)
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),y[tr])
            pb[te]=clf.decision_function(sc.transform(Xte))
        aucs.append(roc_auc_score(y,pb))
    print(f'  {name:24s} n={len(y)} 우울{int(y.sum())} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)
    return len(y),np.mean(aucs),np.std(aucs)
print('=== E-DAIC 증상 앵커 vs 전체 ===',flush=True)
r_all=run('all','전체 인터뷰')
r_anc=run('symptom','증상언급 앵커')
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/edaic_anchor.csv','w') as f:
    f.write('mode,n,AUC,std\n')
    f.write(f'all,{r_all[0]},{r_all[1]:.4f},{r_all[2]:.4f}\n')
    f.write(f'symptom_anchor,{r_anc[0]},{r_anc[1]:.4f},{r_anc[2]:.4f}\n')
print('DONE',flush=True)
