"""exp96 — 시간 경직(rigidity) 마커 4코퍼스 재현. velocity + AR1 + DFA Hurst.
모두 초단위. loss-of-complexity: 우울=AR1↑·Hurst↑·velocity↓ 예상. 4개서 방향 일치?
"""
import numpy as np, warnings, csv, os, re, glob
import pandas as pd
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
W=20; STR=5
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
def logm_vec(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None);L=U@np.diag(np.log(ev))@U.T
    iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def win_logs(seq,d):
    outs=[]
    for s in range(0,len(seq)-W+1,STR):
        seg=seq[s:s+W]
        if seg.std(0).min()<1e-8: seg=seg+np.random.randn(*seg.shape)*1e-6
        C,_=ledoit_wolf(seg);C=C+1e-3*np.eye(d);outs.append(logm_vec(C))
    return np.array(outs)
def dfa(x):
    x=np.asarray(x,float)
    if len(x)<24: return np.nan
    y=np.cumsum(x-x.mean())
    scales=np.unique(np.floor(np.logspace(np.log10(4),np.log10(len(x)//4),10)).astype(int))
    scales=scales[scales>=4];F=[]
    for s in scales:
        n=len(y)//s
        if n<1: continue
        r=[]
        for i in range(n):
            seg=y[i*s:(i+1)*s];t=np.arange(s);fit=np.polyval(np.polyfit(t,seg,1),t)
            r.append(np.sqrt(np.mean((seg-fit)**2)))
        if r: F.append((s,np.mean(r)))
    if len(F)<3: return np.nan
    return float(np.polyfit(np.log([a for a,_ in F]),np.log([b for _,b in F]),1)[0])
def dynamics(seq):
    if len(seq)<24: return None
    mean=seq.mean(0);d2m=np.linalg.norm(seq-mean,axis=1);dstep=np.linalg.norm(np.diff(seq,axis=0),axis=1)
    a,b=d2m[:-1],d2m[1:];ar1=np.corrcoef(a,b)[0,1] if a.std()>1e-9 and b.std()>1e-9 else np.nan
    return [dstep.mean(),ar1,dfa(d2m)]  # velocity, AR1, Hurst
NAMES=['velocity','AR1','Hurst']
def report(corp,DYN,Y):
    DYN=np.array(DYN,float);Y=np.array(Y)
    ok=~np.isnan(DYN).any(1);DYN=DYN[ok];Y=Y[ok]
    print(f'\n#### {corp} (n={len(Y)}, MDD={int(Y.sum())}) ####',flush=True)
    dirs={}
    for k,nm in enumerate(NAMES):
        a=DYN[Y==1,k];b=DYN[Y==0,k]
        try:u,p=mannwhitneyu(a,b,alternative='two-sided')
        except:p=1
        dr='MDD>HC' if a.mean()>b.mean() else 'MDD<HC';dirs[nm]=(dr,p)
        print(f'  {nm:9s} MDD={a.mean():.3f} HC={b.mean():.3f} [{dr}] p={p:.2e}',flush=True)
    return dirs
def persec(df):
    df.columns=[c.strip() for c in df.columns]
    if 'timestamp' not in df or not set(AUS).issubset(df.columns): return None
    if 'success' in df:df=df[df['success']==1]
    if 'confidence' in df:df=df[df['confidence']>0.9]
    if len(df)<50: return None
    df=df.copy();df['sec']=np.floor(df['timestamp']).astype(int)
    return df.groupby('sec')[AUS].mean().values
def proc(series,d):
    if series is None or len(series)<W+4*STR: return None
    X=(series-series.mean(0))/(series.std(0)+1e-6)
    return dynamics(win_logs(X,d))
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/'
alldirs={}
# --- D-Vlog ---
R=Path(DATA+'D-Vlog')
def nf(V):
    T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
raw=[];Yd=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:V=np.load(fv)
    except:continue
    if V.ndim!=2 or V.shape[0]<W+4*STR or V.shape[1]!=136:continue
    raw.append(nf(V));Yd.append(1 if r['label'].strip().lower().startswith('depress') else 0)
pca=PCA(20).fit(np.vstack([raw[i][::5] for i in range(len(raw))]))
DY=[];YY=[]
for i in range(len(Yd)):
    Z=pca.transform(raw[i]);dy=proc(Z,20)
    if dy:DY.append(dy);YY.append(Yd[i])
alldirs['D-Vlog']=report('D-Vlog',DY,YY)
# --- CMDC (Q별 초단위 concat) ---
DY=[];YY=[]
for dfold in sorted(glob.glob(DATA+'CMDC/extracted/*/')):
    name=os.path.basename(dfold.rstrip('/'))
    if not (name.startswith('HC') or name.startswith('MDD')):continue
    segs=[]
    for f in glob.glob(dfold+'Q*.csv'):
        if not re.match(r'^Q\d+\.csv$',os.path.basename(f)):continue
        try:ps=persec(pd.read_csv(f))
        except:ps=None
        if ps is not None and len(ps)>=W:segs.append(ps)
    if not segs:continue
    series=np.vstack(segs);dy=proc(series,17)
    if dy:DY.append(dy);YY.append(1 if name.startswith('MDD') else 0)
alldirs['CMDC']=report('CMDC',DY,YY)
# --- E-DAIC ---
lab={}
for sp in ['train','dev','test']:
    p=DATA+f'E-DAIC/labels/{sp}_split.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            pid=r.get('Participant_ID','').strip()
            b=r.get('PHQ_Binary',r.get('PHQ8_Binary','')).strip()
            if pid and b in ('0','1'):lab[pid]=int(b)
DY=[];YY=[]
for dfold in sorted(glob.glob(DATA+'E-DAIC/extracted/*_P/')):
    pid=os.path.basename(dfold.rstrip('/')).replace('_P','')
    if pid not in lab:continue
    f=dfold+f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'
    if not os.path.exists(f):continue
    try:ps=persec(pd.read_csv(f,usecols=lambda c:c.strip() in set(['timestamp','success','confidence']+AUS)))
    except:ps=None
    dy=proc(ps,17) if ps is not None else None
    if dy:DY.append(dy);YY.append(lab[pid])
alldirs['E-DAIC']=report('E-DAIC',DY,YY)
# --- LMVD ---
def lmlab(i): return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
want=set(['timestamp','success','confidence']+AUS)
DY=[];YY=[]
for f in sorted(glob.glob(DATA+'LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(f))
    if not m:continue
    try:ps=persec(pd.read_csv(f,usecols=lambda c:c.strip() in want))
    except:ps=None
    dy=proc(ps,17) if ps is not None else None
    if dy:DY.append(dy);YY.append(lmlab(int(m.group(1))))
alldirs['LMVD']=report('LMVD',DY,YY)
# ===== 4코퍼스 방향 일치 =====
print('\n===== 4코퍼스 방향 일치 (우울=경직 가설: AR1↑·Hurst↑·velocity↓) =====',flush=True)
for nm in NAMES:
    row=[f"{c}={alldirs[c][nm][0]}(p={alldirs[c][nm][1]:.0e})" for c in ['D-Vlog','CMDC','E-DAIC','LMVD'] if nm in alldirs.get(c,{})]
    print(f'  [{nm}] '+' | '.join(row),flush=True)
print('DONE',flush=True)
