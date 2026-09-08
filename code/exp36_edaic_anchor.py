"""
exp36 — E-DAIC anchored (정정: transcript에 Ellie 표준 질문이 텍스트로 존재).
transcript(Start_Time,End_Time,Text)에서 증상/부정 질문 문구를 substring 매칭 → 그 뒤 8s 창의
OpenFace2.1 AU covariance → Riemannian tangent → logistic. no-anchor(0.563)와 비교.
결과: results/edaic_anchored.csv
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
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
SEEDS=list(range(10)); POST=8.0
# Ellie 표준 증상/부정 질문의 distinctive 문구 (참가자 답변엔 잘 안 나오는 다어절)
NEG=['good night sleep','get a good night','feeling lately','been feeling','diagnosed with depression',
     'feel guilty','last argument','control your temper','your temper','feel down','feeling down',
     'last time you felt really sad','something that makes you really mad','hard time','regret']
SYM=['good night sleep','get a good night','feeling lately','diagnosed with depression','feel guilty',
     'last argument','your temper','feel down','regret','hard time','been feeling']

def labels():
    lab={}
    for f in ['train_split.csv','dev_split.csv','test_split.csv']:
        p=E/'labels'/f
        if p.exists():
            for r in csv.DictReader(open(p)):
                pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
                if b not in(None,''):lab[pid]=int(float(b))
    return lab
def au_series(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
    if not fs:return None,None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try:
        ti=h.index('timestamp');ci=h.index('confidence');oi=h.index('success')
        ai=[h.index(c) for c in AUc];rx,ry,rz=h.index('pose_Rx'),h.index('pose_Ry'),h.index('pose_Rz')
    except:return None,None
    ts,fe=[],[]
    for ln in open(fs[0]).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            if max(abs(float(v[rx])),abs(float(v[ry])),abs(float(v[rz])))>0.35:continue
            ts.append(float(v[ti]));fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(ts),np.array(fe)
def transcript(pid):
    p=E/'extracted'/f'{pid}_P'/f'{pid}_Transcript.csv'
    rows=[]
    if p.exists():
        for r in csv.DictReader(open(p)):
            try: rows.append((float(r['Start_Time']),float(r['End_Time']),(r['Text'] or '').lower()))
            except: pass
    return rows
def build(keys,name):
    lab=labels();C=[];y=[];nseg=[]
    for pid,l in lab.items():
        ts,au=au_series(pid)
        if ts is None or len(ts)<20:continue
        segs=[]
        for st,en,txt in transcript(pid):
            if any(k in txt for k in keys):
                m=(ts>=en)&(ts<en+POST)
                if m.sum()>=6:segs.append(au[m])
        if not segs:continue
        seg=np.vstack(segs)
        if len(seg)<15:continue
        seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);C.append(c);y.append(l);nseg.append(len(segs))
    C=np.array(C);y=np.array(y)
    if len(y)<30:
        print(f'  [{name}] 표본부족 n={len(y)}',flush=True);return
    aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            t=TangentSpace(metric='riemann').fit(C[tr]);Xtr=t.transform(C[tr]);Xte=t.transform(C[te])
            sc=StandardScaler().fit(Xtr)
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),y[tr])
            pb[te]=clf.decision_function(sc.transform(Xte))
        aucs.append(roc_auc_score(y,pb))
    print(f'  [{name}] n={len(y)} 우울{int(y.sum())} 평균seg{np.mean(nseg):.1f} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)
    return name,len(y),int(y.sum()),float(np.mean(aucs)),float(np.std(aucs))

print('=== E-DAIC anchored (transcript 질문 문구 매칭) — no-anchor 0.563 대비 ===',flush=True)
res=[]
r=build(SYM,'symptom/neg anchor'); res.append(r) if r else None
r=build(NEG,'ALL_neg anchor'); res.append(r) if r else None
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/edaic_anchored.csv','w') as f:
    f.write('setting,n,depressed,AUC,std\n')
    for r in res:
        if r: f.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f}\n')
print('DONE',flush=True)
