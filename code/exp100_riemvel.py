"""exp100 — 알고리즘 변형: Riemannian velocity stream.
baseline: SPD상태→GRU. 변형: SPD상태 + affine-invariant 측지속도(Σt^-.5 Σt+1 Σt^-.5 의 log) → GRU.
곡면 위 협응 이동을 명시 모델링. D-Vlog 공식, 다seed. 이기면 algorithmic novelty.
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); dev='cuda'; T=256; W=64; STR=32; EP=60; NSEED=8
d=20; P=d*(d+1)//2
def nf(V):
    T2=V.shape[0];Pp=V.reshape(T2,68,2).astype(float);Pp=Pp-Pp.mean(1,keepdims=True)
    sc=np.sqrt((Pp**2).sum(2).mean(1,keepdims=True))+1e-6;return (Pp/sc[:,None]).reshape(T2,136)
def resamp(X,t=T):
    n=len(X);idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def win_covs(seq):
    cs=[]
    for s in range(0,T-W+1,STR):
        seg=seq[s:s+W];seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);cs.append(c+1e-3*np.eye(d))
    return np.array(cs)
def logm_vec(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None);L=U@np.diag(np.log(ev))@U.T
    iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def inv_sqrt(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None);return U@np.diag(ev**-0.5)@U.T
def riem_vel(covs):  # (K,d,d)->(K,P), 첫스텝 0패드
    out=[np.zeros(P)]
    for t in range(len(covs)-1):
        ish=inv_sqrt(covs[t]);Tt=ish@covs[t+1]@ish;out.append(logm_vec(Tt))
    return np.array(out)
class SPDseq(nn.Module):
    def __init__(self,d,out=16):
        super().__init__();self.W=nn.Parameter(torch.randn(out,d)*0.1);self.out=out
        self.pre=nn.Sequential(nn.Linear(out*(out+1)//2,64),nn.ReLU())
    def sl(self,M):
        I=torch.eye(self.out,device=M.device,dtype=torch.float64)
        Bm=(self.W.double()@M.double()@self.W.double().transpose(0,1))+1e-2*I
        ev,U=torch.linalg.eigh(Bm);ev=ev.clamp(min=1e-3);M2=U@torch.diag_embed(ev)@U.transpose(1,2)
        ev2,U2=torch.linalg.eigh(M2+1e-4*I);L=U2@torch.diag_embed(torch.log(ev2.clamp(min=1e-4)))@U2.transpose(1,2)
        idx=torch.triu_indices(self.out,self.out);return L[:,idx[0],idx[1]].float()
    def forward(self,covs):
        Bs,K,dd,_=covs.shape;v=self.sl(covs.reshape(Bs*K,dd,dd)).reshape(Bs,K,-1);return self.pre(v)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Model(nn.Module):
    def __init__(self,Ca,use_vel):
        super().__init__();self.spd=SPDseq(d);self.use_vel=use_vel
        gin=64
        if use_vel:self.vfc=nn.Sequential(nn.Linear(P,64),nn.ReLU());gin=128
        self.gru=nn.GRU(gin,32,batch_first=True,bidirectional=True);self.a=CNNb(Ca)
        self.cls=nn.Sequential(nn.Linear(64+96,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,vel,xa):
        s=self.spd(covs)
        if self.use_vel:s=torch.cat([s,self.vfc(vel)],2)
        o,_=self.gru(s);return self.cls(torch.cat([o.mean(1),self.a(xa)],1)).squeeze(-1)
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
pca=PCA(d).fit(np.vstack([raw[i][::3] for i in range(len(raw)) if Fd[i]=='train']))
COVS=[];VEL=[];AUD=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);cv=win_covs(resamp(Zz))
    COVS.append(cv);VEL.append(riem_vel(cv))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);AUD.append(resamp(Aa))
COVS=np.array(COVS);VEL=np.array(VEL);AUD=np.array(AUD);Ca=AUD.shape[2]
# velocity 표준화
vm=VEL[Fd=='train'].reshape(-1,P).mean(0);vs=VEL[Fd=='train'].reshape(-1,P).std(0)+1e-6
VEL=(VEL-vm)/vs
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
def run(use_vel,seed):
    torch.manual_seed(seed);net=Model(Ca,use_vel).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Ct=T3(COVS[tr]);Vt=T3(VEL[tr]);At=T3(AUD[tr]).transpose(1,2);yt=T3(Y[tr])
    Cv=T3(COVS[va]);Vv=T3(VEL[va]);Av=T3(AUD[va]).transpose(1,2)
    Ce=T3(COVS[te]);Ve=T3(VEL[te]);Ae=T3(AUD[te]).transpose(1,2)
    n=tr.sum();bs=64;bva=0;bp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Ct[b],Vt[b],At[b]),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():pv=torch.sigmoid(net(Cv,Vv,Av)).cpu().numpy();pt=torch.sigmoid(net(Ce,Ve,Ae)).cpu().numpy()
        try:
            a=roc_auc_score(Y[va],pv)
            if a>bva:bva=a;bp=pt
        except:pass
    return roc_auc_score(Y[te],bp),f1_score(Y[te],(bp>0.5).astype(int))
print(f'=== GRU(baseline) vs GRU+RiemVel (D-Vlog 공식, {NSEED}seed) ===',flush=True)
b=[];v=[];win=0
for s in range(NSEED):
    ab,fb=run(False,s);av,fv2=run(True,s);b.append(ab);v.append(av);win+=int(av>ab)
    print(f'  seed{s}: base={ab:.4f} +vel={av:.4f} {"vel승" if av>ab else "base승"}',flush=True)
print(f'  >> baseline {np.mean(b):.4f}±{np.std(b):.4f} | +RiemVel {np.mean(v):.4f}±{np.std(v):.4f} | vel승률 {win}/{NSEED}',flush=True)
print('DONE',flush=True)
