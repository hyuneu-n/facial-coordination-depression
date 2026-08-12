"""
exp33 — coordination 5종 측정을 DAIC-WOZ(anchor) + E-DAIC(quality gate)에도 (CMDC와 동일).
측정: covariance(Riemannian), lagged correlation, NMF synergy, HMM state-transition, MdRQA.
+ coupling difference heatmap (MDD-HC) 을 DAIC / E-DAIC 에 대해 저장.
결과: results/coordination_DAIC.csv, coordination_EDAIC.csv, G5_coupling_heatmap_DAIC.png, _EDAIC.png
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
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings('ignore')
try:
    from hmmlearn.hmm import GaussianHMM; HAVE_HMM=True
except: HAVE_HMM=False
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); OUT='/home/hyuneun/disk_b/🟡facial-prodrome/results/'
SEEDS=list(range(10))
NAVY='#4472C4'

def f_lagcorr(X,lag=3):
    d=X.shape[1]
    if len(X)<=lag: return np.zeros(d*d)
    A=X[:-lag]; Bb=X[lag:]
    A=(A-A.mean(0))/(A.std(0)+1e-6); Bb=(Bb-Bb.mean(0))/(Bb.std(0)+1e-6)
    return ((A.T@Bb)/len(A)).flatten()
def f_rqa(X):
    Xn=(X-X.mean(0))/(X.std(0)+1e-6)
    D=squareform(pdist(Xn)); thr=np.percentile(D,10); R=(D<thr).astype(int); N=len(R)
    RR=R.sum()/(N*N+1e-9)
    L=[]
    for k in range(-N+1,N):
        d=np.diag(R,k); c=0
        for v in d:
            if v: c+=1
            else:
                if c>=2: L.append(c)
                c=0
        if c>=2: L.append(c)
    tot=R.sum()+1e-9; DET=sum(L)/tot
    ent=0.0
    if L:
        u,cnt=np.unique(L,return_counts=True); p=cnt/cnt.sum(); ent=-(p*np.log(p+1e-12)).sum()
    return np.array([RR,DET,ent,np.mean(L) if L else 0])
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
        m=GaussianHMM(n_components=k,covariance_type='diag',n_iter=20,random_state=0)
        m.fit(X); return m.transmat_.flatten()
    except: return None

def run_dataset(SEQ,y,name):
    subs=list(SEQ.keys()); y=np.array(y)
    def auc_vec(feats):
        X=np.nan_to_num(np.array(feats)); aucs=[]
        for sd in SEEDS:
            skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
            for tr,te in skf.split(X,y):
                clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
                clf.fit(X[tr],y[tr]); pb[te]=clf.decision_function(X[te])
            aucs.append(roc_auc_score(y,pb))
        return np.mean(aucs),np.std(aucs)
    res={}
    mats=np.array([ (lambda X:(ledoit_wolf(X)[0]))(SEQ[s]) for s in subs ]); aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y))
        for tr,te in skf.split(mats,y):
            clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(mats[tr],y[tr]); pb[te]=clf.decision_function(mats[te])
        aucs.append(roc_auc_score(y,pb))
    res['covariance']=(np.mean(aucs),np.std(aucs))
    res['lagged_corr']=auc_vec([f_lagcorr(SEQ[s]) for s in subs])
    res['NMF']=auc_vec([f_nmf(SEQ[s]) for s in subs])
    hmmf=[f_hmm(SEQ[s]) for s in subs]
    if all(h is not None for h in hmmf) and len(set(len(h) for h in hmmf))==1:
        res['HMM']=auc_vec(hmmf)
    else:
        res['HMM']=(np.nan,np.nan)
    res['MdRQA']=auc_vec([f_rqa(SEQ[s]) for s in subs])
    print(f'\n=== {name} coordination (n={len(subs)}, MDD{int(y.sum())}) ===',flush=True)
    for k in ['covariance','lagged_corr','NMF','HMM','MdRQA']:
        m,s=res[k]; print(f'  {k:14s} AUC={m:.3f}±{s:.3f}' if not np.isnan(m) else f'  {k:14s} n/a',flush=True)
    return subs,y,mats,res

def heatmap(mats,y,AUnames,title,path):
    mdd=np.array(mats)[y==1].mean(0); hc=np.array(mats)[y==0].mean(0); diff=mdd-hc; np.fill_diagonal(diff,0)
    fig,ax=plt.subplots(figsize=(6.4,5.4)); vmax=np.abs(diff).max()
    im=ax.imshow(diff,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(len(AUnames)));ax.set_yticks(range(len(AUnames)))
    ax.set_xticklabels(AUnames,rotation=90,fontsize=7);ax.set_yticklabels(AUnames,fontsize=7)
    cb=plt.colorbar(im,fraction=0.046,pad=0.04);cb.set_label('Δ coupling (MDD − HC)',fontsize=9)
    ax.set_title(title,fontsize=11); ax.grid(False)
    plt.tight_layout(); plt.savefig(path); plt.close()
    iu=np.triu_indices(len(AUnames),1); mags=np.abs(diff[iu]); order=np.argsort(mags)[::-1][:6]
    print(f'  top pairs [{title}]:',[f'{AUnames[iu[0][o]]}-{AUnames[iu[1][o]]}:{diff[iu[0][o],iu[1][o]]:+.2f}' for o in order],flush=True)

# ===== DAIC-WOZ anchor =====
AUd=['AU01','AU02','AU04','AU05','AU06','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU25','AU26']
AUdc=[a+'_r' for a in AUd]
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
NEG=['feel_lately','depression_diagnosed','feelguilty','regret','feelbadly','last_argument','control_temper']
def dseries(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None,None
    h=[x.strip() for x in open(p).readline().split(',')];ti,oi=h.index('timestamp'),h.index('success');ai=[h.index(c) for c in AUdc]
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
SEQd={};yd=[]
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
    SEQd[pid]=seg;yd.append(l)
subs,yd2,matsd,resd=run_dataset(SEQd,yd,'DAIC-WOZ(anchor)')
heatmap(matsd,yd2,AUd,'Inter-AU coupling difference (DAIC-WOZ anchor)',OUT+'G5_coupling_heatmap_DAIC.png')
with open(OUT+'coordination_DAIC.csv','w') as f:
    f.write('measure,AUC,std\n')
    for k in ['covariance','lagged_corr','NMF','HMM','MdRQA']:f.write(f'{k},{resd[k][0]:.4f},{resd[k][1]:.4f}\n')

# ===== E-DAIC quality gate =====
AUe=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45']
AUec=[a+'_r' for a in AUe]
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
SEQe={};ye=[]
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
    seg=np.array(seg);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);SEQe[pid]=seg;ye.append(l)
subs,ye2,matse,rese=run_dataset(SEQe,ye,'E-DAIC(quality gate)')
heatmap(matse,ye2,AUe,'Inter-AU coupling difference (E-DAIC quality gate)',OUT+'G5_coupling_heatmap_EDAIC.png')
with open(OUT+'coordination_EDAIC.csv','w') as f:
    f.write('measure,AUC,std\n')
    for k in ['covariance','lagged_corr','NMF','HMM','MdRQA']:f.write(f'{k},{rese[k][0]:.4f},{rese[k][1]:.4f}\n')
print('\nDONE → coordination_DAIC.csv / coordination_EDAIC.csv / heatmaps',flush=True)
