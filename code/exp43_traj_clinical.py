"""
exp43 — SPD 궤적 기하를 임상(CMDC/DAIC AU)서 검증.
anchor 구간 AU → segment별 sliding-window cov → 궤적. velocity/dispersion(우울↓?) + static에 더해지나.
결과: results/exp43_traj_clinical.csv
"""
import numpy as np, warnings, csv, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=list(range(10)); WIN,STEP=20,5
def seg_covs(segs,d):
    per=[]
    for s in segs:
        s=(s-s.mean(0))/(s.std(0)+1e-6)
        cs=[]
        for a in range(0,len(s)-WIN+1,STEP):
            c,_=ledoit_wolf(s[a:a+WIN]); cs.append(c+1e-4*np.eye(d))
        if len(cs)>=2: per.append(cs)
    return per
def traj_feats(per):
    allc=[c for cs in per for c in cs]
    if len(allc)<4: return None,None
    M=mean_riemann(np.array(allc))
    disp=np.mean([distance_riemann(c,M) for c in allc])
    steps=[]
    for cs in per:
        steps+=[distance_riemann(cs[i],cs[i+1]) for i in range(len(cs)-1)]
    vel=np.mean(steps); path=np.sum(steps); acc=np.std(steps)
    return M,np.array([path,vel,disp,acc,np.log(len(allc))])

def evalset(subj_segs, y, d, name):
    M=[];T=[];yy=[]
    for i,segs in enumerate(subj_segs):
        per=seg_covs(segs,d); m,t=traj_feats(per)
        if m is None: continue
        M.append(m);T.append(t);yy.append(y[i])
    M=np.array(M);T=np.nan_to_num(np.array(T));yy=np.array(yy)
    print(f'\n=== {name} n={len(yy)}(dep{yy.sum()}) ===',flush=True)
    for j,nm in enumerate(['path','velocity','dispersion','accel']):
        md,hc=T[yy==1,j].mean(),T[yy==0,j].mean()
        print(f'  {nm:10s} MDD={md:.3f} HC={hc:.3f} ({"↓우울" if md<hc else "↑우울"})',flush=True)
    def cv(kind):
        a=[]
        for s in SEEDS:
            skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(yy))
            for tr,te in skf.split(T,yy):
                ts=TangentSpace(metric='riemann').fit(M[tr]);Xs=ts.transform(M)
                if kind=='stat':F=Xs
                elif kind=='traj':F=T
                else:F=np.column_stack([Xs,T])
                sc=StandardScaler().fit(F[tr]);cl=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(F[tr]),yy[tr])
                pb[te]=cl.decision_function(sc.transform(F[te]))
            a.append(roc_auc_score(yy,pb))
        return np.mean(a),np.std(a)
    r={k:cv(k) for k in ['stat','traj','comb']}
    print(f'  정적 {r["stat"][0]:.3f} | 궤적 {r["traj"][0]:.3f} | 결합 {r["comb"][0]:.3f}',flush=True)
    return name,r

# CMDC
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
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if len(fe)>=WIN else None
segsC=[];yC=[]
for s,l in cl.items():
    ss=[cq(s,q) for q in [3,7]]; ss=[x for x in ss if x is not None]
    if ss: segsC.append(ss);yC.append(l)
rC=evalset(segsC,yC,len(AUc),'CMDC')

# DAIC
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
NEG=['feel_lately','depression_diagnosed','feelguilty','regret','feelbadly','last_argument','control_temper']
def dser(pid):
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
def dtr(pid):
    p=D/f'{pid}_TRANSCRIPT.csv';r_=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try:r_.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except:pass
    return r_
segsD=[];yD=[]
for pid,l in dl.items():
    ts,au=dser(pid)
    if ts is None:continue
    wins=[au[(ts>=sp)&(ts<sp+8.0)] for st,sp,spk,val in dtr(pid) if spk=='Ellie' and any(val.startswith(t) for t in NEG)]
    wins=[w for w in wins if len(w)>=WIN]
    if wins: segsD.append(wins);yD.append(l)
rD=evalset(segsD,yD,len(AUd),'DAIC')

with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp43_traj_clinical.csv','w') as f:
    f.write('dataset,static,trajectory,combined\n')
    for nm,r in [rC,rD]: f.write(f'{nm},{r["stat"][0]:.4f},{r["traj"][0]:.4f},{r["comb"][0]:.4f}\n')
print('\nDONE → exp43_traj_clinical.csv',flush=True)
