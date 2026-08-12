"""
exp34 — (CPU) B2 + D2.
B2: CMDC 질문별 anchor 근거 = 각 Q 단독 anchor covariance-Riemannian AUC → 왜 Q3/Q7?
D2: 주요 분류 결과(CMDC/DAIC/E-DAIC) effect size(Cliff's delta) + bootstrap 95% CI.
결과: results/cmdc_perQ.csv, results/stats_effectsize.csv
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
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); OUT='/home/hyuneun/disk_b/🟡facial-prodrome/results/'
SEEDS=list(range(10))
def cliffs_delta(a,b):  # P(a>b)-P(a<b), a=score(dep), b=score(hc)
    a=np.asarray(a);b=np.asarray(b);gt=0;lt=0
    for x in a:
        gt+=np.sum(x>b);lt+=np.sum(x<b)
    n=len(a)*len(b); return (gt-lt)/n if n else 0.0
def cv_scores(C,y):
    C=np.array(C);y=np.array(y);pb_all=np.zeros(len(y))
    # 대표 1 seed로 per-subject decision score (effect size/CI용)
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    for tr,te in skf.split(C,y):
        t=TangentSpace(metric='riemann').fit(C[tr]);Xtr=t.transform(C[tr]);Xte=t.transform(C[te])
        sc=StandardScaler().fit(Xtr)
        clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),y[tr])
        pb_all[te]=clf.decision_function(sc.transform(Xte))
    return pb_all
def auc_10seed(C,y):
    C=np.array(C);y=np.array(y);a=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            t=TangentSpace(metric='riemann').fit(C[tr]);Xtr=t.transform(C[tr]);Xte=t.transform(C[te])
            sc=StandardScaler().fit(Xtr)
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),y[tr])
            pb[te]=clf.decision_function(sc.transform(Xte))
        a.append(roc_auc_score(y,pb))
    return np.array(a)
def bootstrap_ci(y,score,n=2000):
    y=np.array(y);score=np.array(score);rng=np.random.RandomState(0);aucs=[]
    idx=np.arange(len(y))
    for _ in range(n):
        bi=rng.choice(idx,len(idx),replace=True)
        if len(np.unique(y[bi]))<2: continue
        aucs.append(roc_auc_score(y[bi],score[bi]))
    return np.percentile(aucs,2.5),np.percentile(aucs,97.5)

# ===================== CMDC =====================
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if len(fe)>=10 else None
# B2: 질문별 단독 anchor
QTXT={1:'식욕/체중',2:'수면 상세',3:'수면',4:'집중',5:'피로/에너지',6:'죄책/무가치',7:'피로',8:'정신운동',9:'자해생각',10:'기분',11:'흥미',12:'종합'}
print('=== B2: CMDC 질문별 단독 anchor AUC (왜 Q3+Q7?) ===',flush=True)
perQ=[]
for q in range(1,13):
    Cq=[];yq=[]
    for s,l in cl.items():
        seg=cq(s,q)
        if seg is not None and len(seg)>=10:
            seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);Cq.append(c);yq.append(l)
    if len(yq)<30 or sum(yq)<8:
        print(f'  Q{q:<2} n={len(yq)} 표본부족',flush=True); continue
    a=auc_10seed(Cq,yq); perQ.append((q,len(yq),int(sum(yq)),np.mean(a),np.std(a)))
    print(f'  Q{q:<2} ({QTXT.get(q,"")}) n={len(yq)} AUC={np.mean(a):.3f}±{np.std(a):.3f}',flush=True)
perQ.sort(key=lambda r:-r[3])
print('  → 상위:',[f'Q{q}:{au:.3f}' for q,_,_,au,_ in perQ[:4]],flush=True)
with open(OUT+'cmdc_perQ.csv','w') as f:
    f.write('Q,desc,n,depressed,AUC,std\n')
    for q,n,d,au,sd in sorted(perQ,key=lambda r:r[0]): f.write(f'{q},{QTXT.get(q,"")},{n},{d},{au:.4f},{sd:.4f}\n')

# 데이터셋별 covariance 세트 구성 (D2용)
def cmdc_anchor():
    Cc=[];yc=[]
    for s,l in cl.items():
        segs=[cq(s,q) for q in [3,7]]
        if all(x is not None for x in segs):
            seg=np.vstack(segs);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);Cc.append(c);yc.append(l)
    return Cc,yc

# ===================== DAIC =====================
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
NEG=['feel_lately','depression_diagnosed','feelguilty','regret','feelbadly','last_argument','control_temper']
def dseries(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None,None
    h=[x.strip() for x in open(p).readline().split(',')];ti,oi=h.index('timestamp'),h.index('success');ai=[h.index(c) for c in AUd]
    ts,fe=[],[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            ts.append(float(v[ti]));fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(ts),np.array(fe)
def dtrans(pid):
    p=D/f'{pid}_TRANSCRIPT.csv';rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try:rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except:pass
    return rows
def daic_anchor():
    Cd=[];yd=[]
    for pid,l in dl.items():
        ts,au=dseries(pid)
        if ts is None:continue
        segs=[]
        for st,sp,spk,val in dtrans(pid):
            if spk=='Ellie' and any(val.startswith(t) for t in NEG):
                m=(ts>=sp)&(ts<sp+8.0)
                if m.sum()>=8:segs.append(au[m])
        if not segs:continue
        seg=np.vstack(segs);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6)
        if len(seg)<15:continue
        c,_=ledoit_wolf(seg);Cd.append(c);yd.append(l)
    return Cd,yd

# ===================== E-DAIC =====================
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
def edaic_gate():
    Ce=[];ye=[]
    for pid,l in el.items():
        fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
        if not fs:continue
        h=[x.strip() for x in open(fs[0]).readline().split(',')]
        try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc];rx,ry,rz=h.index('pose_Rx'),h.index('pose_Ry'),h.index('pose_Rz')
        except:continue
        seg=[]
        for ln in open(fs[0]).readlines()[1:]:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
                if max(abs(float(v[rx])),abs(float(v[ry])),abs(float(v[rz])))>0.35:continue
                seg.append([float(v[i]) for i in ai])
            except:pass
        if len(seg)<20:continue
        seg=np.array(seg);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);Ce.append(c);ye.append(l)
    return Ce,ye

print('\n=== D2: effect size(Cliff\'s delta) + bootstrap 95% CI ===',flush=True)
out=[]
for name,fn in [('CMDC',cmdc_anchor),('DAIC-WOZ',daic_anchor),('E-DAIC',edaic_gate)]:
    C,y=fn();y=np.array(y)
    a=auc_10seed(C,y); score=cv_scores(C,y)
    delta=cliffs_delta(score[y==1],score[y==0]); lo,hi=bootstrap_ci(y,score)
    mag='negligible' if abs(delta)<0.147 else ('small' if abs(delta)<0.33 else ('medium' if abs(delta)<0.474 else 'large'))
    print(f'  {name:9s} AUC={np.mean(a):.3f} [95%CI {lo:.3f}-{hi:.3f}] Cliff δ={delta:+.3f}({mag})',flush=True)
    out.append((name,len(y),int(y.sum()),np.mean(a),lo,hi,delta,mag))
with open(OUT+'stats_effectsize.csv','w') as f:
    f.write('dataset,n,depressed,AUC,CI_lo,CI_hi,cliffs_delta,magnitude\n')
    for r in out: f.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f},{r[7]}\n')
print('\nDONE → cmdc_perQ.csv / stats_effectsize.csv',flush=True)
