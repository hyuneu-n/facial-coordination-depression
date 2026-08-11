"""
exp7 — 모든 신호 결합 (긍정앵커, DAIC). 지표는 딱 2개: AUC + p값.
신호군:
  A. AU 통계 (평균+std)
  B. 저차원성 (top1, participation, eff_rank)  ← 선형대수
  C. head-motion 동역학 (pose Rx,Ry,Rz 움직임/고정도)  ← 네 강점(자세)
  D. gaze (시선)
점진 결합: A → +B → +C → +D. 어느 조합이 최고인지.
"""
import numpy as np, csv, warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
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
def read_txt(pid,suffix,cols):
    p=D/f'{pid}_CLNF_{suffix}.txt'
    if not p.exists(): return None,None
    h=[x.strip() for x in open(p).readline().split(',')]
    try: ti,oi=h.index('timestamp'),h.index('success'); ci=[h.index(c) for c in cols]
    except: return None,None
    ts,fe=[],[]
    with open(p) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                ts.append(float(v[ti])); fe.append([float(v[i]) for i in ci])
            except: pass
    return np.array(ts),np.array(fe)

def seg_of(ts,arr,anch):
    segs=[arr[(ts>=a)&(ts<a+POST)] for a in anch]
    segs=[s for s in segs if len(s)>=4]
    return np.vstack(segs) if segs else None

def lowdim(seg):
    X=seg-seg.mean(0,keepdims=True)
    s=np.linalg.svd(X,compute_uv=False); s=s[s>1e-8]
    if len(s)<2: return np.array([0,0,0.])
    p=s/s.sum(); top1=p[0]; ent=-(p*np.log(p+1e-12)).sum(); eff=np.exp(ent)
    s2=s**2; part=(s2.sum()**2)/((s2**2).sum())
    return np.array([top1,part,eff])

def headmotion(seg):
    # seg: (T,3) pose 회전. 움직임량(고개 흔듦) + 고정도
    if len(seg)<2: return np.array([0,0,0,0,0,0.])
    d=np.abs(np.diff(seg,axis=0))
    return np.concatenate([seg.std(0), d.mean(0)])  # 자세 변동성 + 움직임속도 (6dim)

lab=labels()
FA,FB,FC,FD,y=[],[],[],[],[]
for pid,l in lab.items():
    ts,au=read_txt(pid,'AUs',AU)
    if ts is None: continue
    anch=anchors(transcript(pid),POS)
    sa=seg_of(ts,au,anch)
    if sa is None: continue
    base=au.mean(0); bs=au.std(0)+1e-6; san=(sa-base)/bs
    # A: AU 통계
    fa=np.concatenate([san.mean(0),san.std(0)])
    # B: 저차원성
    fb=lowdim(san)
    # C: head motion
    tp,po=read_txt(pid,'pose',['Rx','Ry','Rz'])
    fc=headmotion(seg_of(tp,po,anch)) if tp is not None else np.zeros(6)
    # D: gaze
    tg,gz=read_txt(pid,'gaze',['x_0','y_0','z_0'])
    sg=seg_of(tg,gz,anch)
    fd=np.concatenate([sg.mean(0),sg.std(0)]) if sg is not None else np.zeros(6)
    FA.append(fa);FB.append(fb);FC.append(fc);FD.append(fd);y.append(l)
FA,FB,FC,FD,y=map(np.array,(FA,FB,FC,FD,y))
print(f'n={len(y)}, 우울{int(y.sum())}/정상{int((y==0).sum())}\n',flush=True)

def auc_p(X,y):
    aucs=[]
    for s in SEEDS:
        cv=StratifiedKFold(5,shuffle=True,random_state=s)
        aucs.append(cross_val_score(LogisticRegression(max_iter=2000,class_weight='balanced'),X,y,cv=cv,scoring='roc_auc').mean())
    # 순열검정 p값 (앵커효과 유의미?): 라벨섞어 100회
    obs=np.mean(aucs); null=[]
    rng=np.random.RandomState(0)
    for _ in range(200):
        yp=rng.permutation(y)
        cv=StratifiedKFold(5,shuffle=True,random_state=42)
        null.append(cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'),X,yp,cv=cv,scoring='roc_auc').mean())
    p=(np.sum(np.array(null)>=obs)+1)/(len(null)+1)
    return obs,np.std(aucs),p

print(f"{'조합':>18} {'AUC':>14} {'p(순열)':>8}",flush=True)
print('-'*44,flush=True)
combos=[('A: AU통계',FA),('A+B(저차원)',np.c_[FA,FB]),('A+B+C(head)',np.c_[FA,FB,FC]),
        ('A+B+C+D(gaze)',np.c_[FA,FB,FC,FD])]
for nm,X in combos:
    Xs=StandardScaler().fit_transform(X)
    a,sd,p=auc_p(Xs,y)
    star='**' if p<0.05 else '*' if p<0.1 else ''
    print(f"{nm:>18} {a:.3f}±{sd:.3f} {p:.3f}{star}",flush=True)
