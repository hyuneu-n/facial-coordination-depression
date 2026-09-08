"""exp103 — 크로스도메인 이식: ROCKET(시계열분류) → 얼굴 협응 우울.
랜덤 conv커널 + PPV/max 풀링 + 로지스틱. 초경량. baseline=SPD+GRU(~0.796).
visual(landmark PCA20)·audio(25) 시계열에 ROCKET. ROCKET단독 / +SPD 결합 비교. D-Vlog 공식.
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn.functional as Fnn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); dev='cuda'; T=256
def nf(V):
    T2=V.shape[0];Pp=V.reshape(T2,68,2).astype(float);Pp=Pp-Pp.mean(1,keepdims=True)
    sc=np.sqrt((Pp**2).sum(2).mean(1,keepdims=True))+1e-6;return (Pp/sc[:,None]).reshape(T2,136)
def resamp(X,t=T):
    n=len(X);idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def rocket_feats(X,n_kernels=500,seed=0):
    # X:(N,C,T) → (N, 2*n_kernels). 랜덤 커널 conv + PPV/max
    rng=np.random.default_rng(seed);Xt=torch.tensor(X,dtype=torch.float32,device=dev);N,C,Tt=X.shape
    feats=[]
    for k in range(n_kernels):
        L=int(rng.choice([7,9,11]));w=rng.standard_normal(L);w=w-w.mean()
        b=float(rng.uniform(-1,1))
        maxd=max(1,int(np.log2((Tt-1)/(L-1))));dil=int(2**rng.integers(0,maxd+1));ch=int(rng.integers(0,C))
        pad=((L-1)*dil)//2 if rng.random()<0.5 else 0
        ker=torch.tensor(w,dtype=torch.float32,device=dev).view(1,1,L)
        conv=Fnn.conv1d(Xt[:,ch:ch+1,:],ker,dilation=dil,padding=pad).squeeze(1)  # (N,T')
        ppv=(conv>b).float().mean(1);mx=conv.max(1)[0]
        feats.append(ppv.cpu().numpy());feats.append(mx.cpu().numpy())
    return np.stack(feats,1)
# 로드
rows=list(csv.DictReader(open(R/'labels.csv')))
raw=[];araw=[];Y=[];Fd=[]
for r in rows:
    idx=r['index'].strip();fold=r.get('fold','').strip().lower()
    if 'val' in fold:fold='valid'
    if fold not in('train','valid','test'):continue
    fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
    if not(fv.exists() and fa.exists()):continue
    try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
    except:continue
    if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136 or Au.ndim!=2 or len(Au)<10:continue
    raw.append(nf(V));araw.append(Au);Y.append(1 if r['label'].strip().lower().startswith('depress') else 0);Fd.append(fold)
Y=np.array(Y);Fd=np.array(Fd)
pca=PCA(20).fit(np.vstack([raw[i][::3] for i in range(len(raw)) if Fd[i]=='train']))
Vis=[];Aud=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);Vis.append(resamp(Zz).T)  # (20,T)
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);Aud.append(resamp(Aa).T)          # (25,T)
Vis=np.array(Vis);Aud=np.array(Aud)
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
def evalfeat(Fmat,name):
    aucs=[];f1s=[]
    for seed in range(5):  # ROCKET 랜덤성 위해 여러 seed
        pass
    sc=StandardScaler().fit(Fmat[tr]);Xt=sc.transform(Fmat[tr]);Xv=sc.transform(Fmat[va]);Xe=sc.transform(Fmat[te])
    best=0;bp=None
    for Cr in [0.01,0.05,0.1,0.5,1.0]:
        m=LogisticRegression(max_iter=3000,C=Cr).fit(Xt,Y[tr])
        a=roc_auc_score(Y[va],m.predict_proba(Xv)[:,1])
        if a>best:best=a;bp=m.predict_proba(Xe)[:,1]
    print(f'  {name:28s} test AUC={roc_auc_score(Y[te],bp):.4f} F1={f1_score(Y[te],(bp>0.5).astype(int)):.4f}',flush=True)
    return roc_auc_score(Y[te],bp)
print('=== ROCKET 이식 (D-Vlog 공식) — 5seed 앙상블 feature ===',flush=True)
# 여러 seed ROCKET feature 이어붙임(앙상블)
RV=np.hstack([rocket_feats(Vis,400,s) for s in range(3)])
RA=np.hstack([rocket_feats(Aud,400,s+10) for s in range(3)])
print(f'  (feature dim: visual {RV.shape[1]}, audio {RA.shape[1]})',flush=True)
evalfeat(RV,'ROCKET-visual only')
evalfeat(RA,'ROCKET-audio only')
evalfeat(np.hstack([RV,RA]),'ROCKET visual+audio')
print('\n[비교] SPD+GRU+audio ~0.796 (exp98). ROCKET이 근접/초과하면 크로스도메인 경량 이식 성공',flush=True)
print('DONE',flush=True)
