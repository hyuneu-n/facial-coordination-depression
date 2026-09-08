"""exp94 — 원리적 coordination 시간 동역학 + D-Vlog·LMVD 재현.
feature: velocity(불안정), AR1(경직/지속), innovation(변동), complexity(스펙트럴 엔트로피=복잡도).
loss-of-complexity 이론: 우울=경직↑(AR1↑)·velocity↓·복잡도↓ 예상. 두 코퍼스 방향 일치?
"""
import numpy as np, warnings, csv, os, re, glob
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
W=30; STR=15
def logm_vec(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None)
    L=U@np.diag(np.log(ev))@U.T;iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def win_logs(seq,d):
    outs=[]
    for s in range(0,len(seq)-W+1,STR):
        seg=seq[s:s+W]
        if seg.std(0).min()<1e-8: seg=seg+np.random.randn(*seg.shape)*1e-6
        C,_=ledoit_wolf(seg);C=C+1e-3*np.eye(d);outs.append(logm_vec(C))
    return np.array(outs)
def spectral_entropy(x):
    x=x-x.mean()
    if x.std()<1e-9 or len(x)<4: return 0.0
    ps=np.abs(np.fft.rfft(x))**2;ps=ps[1:]  # DC 제외
    if ps.sum()<1e-12: return 0.0
    p=ps/ps.sum();p=p[p>0];return float(-(p*np.log(p)).sum()/np.log(len(p)+1e-9))
def dynamics(seq):  # seq:(K,P) tangent 시퀀스 → 4 feature
    if len(seq)<4: return None,None
    mean=seq.mean(0);dist2mean=np.linalg.norm(seq-mean,axis=1)  # 시간별 baseline 이탈
    dstep=np.linalg.norm(np.diff(seq,axis=0),axis=1)            # 연속 변화
    velocity=dstep.mean()
    innovation=dstep.std()
    # AR1 지속성: dist2mean 시계열의 lag-1 자기상관
    a=dist2mean[:-1];b=dist2mean[1:]
    ar1=np.corrcoef(a,b)[0,1] if a.std()>1e-9 and b.std()>1e-9 else 0.0
    complexity=spectral_entropy(dist2mean)
    return seq.mean(0),[velocity,ar1,innovation,complexity]
def cv_auc(X,Y):
    skf=StratifiedKFold(5,shuffle=True,random_state=0);aucs=[]
    for tr,te in skf.split(X,Y):
        sc=StandardScaler().fit(X[tr])
        m=LogisticRegression(max_iter=2000,C=0.5).fit(sc.transform(X[tr]),Y[tr])
        aucs.append(roc_auc_score(Y[te],m.predict_proba(sc.transform(X[te]))[:,1]))
    return np.mean(aucs),np.std(aucs)
NAMES=['velocity','AR1(경직)','innovation','complexity']
def report(corp,STAT,DYN,Y):
    STAT=np.array(STAT);DYN=np.array(DYN);Y=np.array(Y)
    print(f'\n########## {corp}  (n={len(Y)}, MDD={int(Y.sum())}) ##########',flush=True)
    print('--- feature 그룹차이 (MDD vs HC) ---',flush=True)
    for k,nm in enumerate(NAMES):
        a=DYN[Y==1,k];b=DYN[Y==0,k]
        try:u,p=mannwhitneyu(a,b,alternative='two-sided')
        except:p=1
        arrow='MDD<HC' if a.mean()<b.mean() else 'MDD>HC'
        print(f'  {nm:12s} MDD={a.mean():.3f} HC={b.mean():.3f} [{arrow}] p={p:.2e}',flush=True)
    for nm,X in [('정적',STAT),('동역학(4)',DYN),('결합',np.hstack([STAT,DYN]))]:
        m,s=cv_auc(X,Y);print(f'  AUC {nm:10s}= {m:.3f} ± {s:.3f}',flush=True)

# ===== D-Vlog (landmark PCA20) =====
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
def nf(V):
    T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
raw=[];Yd=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:V=np.load(fv)
    except:continue
    if V.ndim!=2 or V.shape[0]<W+2*STR or V.shape[1]!=136:continue
    raw.append(nf(V));Yd.append(1 if r['label'].strip().lower().startswith('depress') else 0)
pca=PCA(20).fit(np.vstack([raw[i][::5] for i in range(len(raw))]))
S1=[];D1=[];Y1=[]
for i in range(len(Yd)):
    Z=pca.transform(raw[i]);Z=(Z-Z.mean(0))/(Z.std(0)+1e-6)
    st,dy=dynamics(win_logs(Z,20))
    if dy is None:continue
    S1.append(st);D1.append(dy);Y1.append(Yd[i])
report('D-Vlog',S1,D1,Y1)

# ===== LMVD (AU 17 직접) =====
def lab(i): return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
S2=[];D2=[];Y2=[]
for f in sorted(glob.glob('/home/hyuneun/disk_b/🟡facial-prodrome/data/LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(f))
    if not m:continue
    i=int(m.group(1))
    try:df=__import__('pandas').read_csv(f)
    except:continue
    df.columns=[c.strip() for c in df.columns]
    au=[c for c in df.columns if re.match(r'AU\d+_r$',c)]
    if len(au)<10:continue
    X=df[au].values.astype(float)
    if len(X)<W+2*STR:continue
    X=(X-X.mean(0))/(X.std(0)+1e-6)
    st,dy=dynamics(win_logs(X,len(au)))
    if dy is None:continue
    S2.append(st);D2.append(dy);Y2.append(lab(i))
if len(set(Y2))>1: report('LMVD',S2,D2,Y2)
else: print('LMVD: 라벨 부족')
print('\nDONE',flush=True)
