"""
exp33b — E-DAIC coordination, MdRQA 제외(전체 녹화 O(N^2) 불가) 빠른 버전.
covariance(Riemannian) / lagged_corr / NMF / HMM 4종 + coupling heatmap.
결과: results/coordination_EDAIC.csv (MdRQA=nan), G5_coupling_heatmap_EDAIC.png
"""
import numpy as np, warnings, csv, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
try:
    from hmmlearn.hmm import GaussianHMM; HAVE_HMM=True
except: HAVE_HMM=False
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); OUT='/home/hyuneun/disk_b/🟡facial-prodrome/results/'
SEEDS=list(range(10))
def f_lagcorr(X,lag=3):
    d=X.shape[1]
    if len(X)<=lag: return np.zeros(d*d)
    A=X[:-lag]; Bb=X[lag:]
    A=(A-A.mean(0))/(A.std(0)+1e-6); Bb=(Bb-Bb.mean(0))/(Bb.std(0)+1e-6)
    return ((A.T@Bb)/len(A)).flatten()
def f_nmf(X,k=4):
    Xp=np.clip(X-X.min(0),0,None)
    try:
        m=NMF(n_components=k,init='nndsvda',max_iter=200,random_state=0)
        W=m.fit_transform(Xp); H=m.components_
        return np.concatenate([[m.reconstruction_err_,(H<0.01).mean()],W.mean(0),W.std(0)])
    except: return np.zeros(2+2*k)
def f_hmm(X,k=3):
    if not HAVE_HMM or len(X)<k+2: return None
    try:
        m=GaussianHMM(n_components=k,covariance_type='diag',n_iter=15,random_state=0); m.fit(X); return m.transmat_.flatten()
    except: return None
AUe=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45']
AUec=[a+'_r' for a in AUe]
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
SEQ={};y=[]
for pid,l in el.items():
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
    if not fs:continue
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUec];rx,ry,rz=h.index('pose_Rx'),h.index('pose_Ry'),h.index('pose_Rz')
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
    seg=np.array(seg);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);SEQ[pid]=seg;y.append(l)
subs=list(SEQ.keys());y=np.array(y)
print(f'E-DAIC gate: n={len(subs)} MDD{int(y.sum())}',flush=True)
def auc_vec(feats):
    X=np.nan_to_num(np.array(feats));a=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
        for tr,te in skf.split(X,y):
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(X[tr],y[tr]);pb[te]=clf.decision_function(X[te])
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
res={}
mats=np.array([ledoit_wolf(SEQ[s])[0] for s in subs]);a=[]
for sd in SEEDS:
    skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
    for tr,te in skf.split(mats,y):
        clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
        clf.fit(mats[tr],y[tr]);pb[te]=clf.decision_function(mats[te])
    a.append(roc_auc_score(y,pb))
res['covariance']=(np.mean(a),np.std(a))
res['lagged_corr']=auc_vec([f_lagcorr(SEQ[s]) for s in subs])
res['NMF']=auc_vec([f_nmf(SEQ[s]) for s in subs])
hmmf=[f_hmm(SEQ[s]) for s in subs]
res['HMM']=auc_vec(hmmf) if all(h is not None for h in hmmf) and len(set(len(h) for h in hmmf))==1 else (np.nan,np.nan)
res['MdRQA']=(np.nan,np.nan)  # 전체 녹화(~2.5만 프레임)엔 O(N^2)라 계산 불가
print('=== E-DAIC coordination (MdRQA 제외) ===',flush=True)
for k in ['covariance','lagged_corr','NMF','HMM','MdRQA']:
    m,s=res[k]; print(f'  {k:14s} '+(f'AUC={m:.3f}±{s:.3f}' if not np.isnan(m) else 'n/a'),flush=True)
with open(OUT+'coordination_EDAIC.csv','w') as f:
    f.write('measure,AUC,std\n')
    for k in ['covariance','lagged_corr','NMF','HMM','MdRQA']:
        m,s=res[k]; f.write(f'{k},{"" if np.isnan(m) else f"{m:.4f}"},{"" if np.isnan(s) else f"{s:.4f}"}\n')
# heatmap
mdd=mats[y==1].mean(0);hc=mats[y==0].mean(0);diff=mdd-hc;np.fill_diagonal(diff,0)
fig,ax=plt.subplots(figsize=(6.4,5.4));vmax=np.abs(diff).max()
im=ax.imshow(diff,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
ax.set_xticks(range(len(AUe)));ax.set_yticks(range(len(AUe)))
ax.set_xticklabels(AUe,rotation=90,fontsize=7);ax.set_yticklabels(AUe,fontsize=7)
cb=plt.colorbar(im,fraction=0.046,pad=0.04);cb.set_label('Δ coupling (MDD − HC)',fontsize=9)
ax.set_title('Inter-AU coupling difference (E-DAIC quality gate)',fontsize=11);ax.grid(False)
plt.tight_layout();plt.savefig(OUT+'G5_coupling_heatmap_EDAIC.png');plt.close()
print('DONE → coordination_EDAIC.csv + heatmap',flush=True)
