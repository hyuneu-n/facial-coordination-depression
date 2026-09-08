"""
exp87 — 성능 강화: SPD-manifold + TCN + audio 3-stream 결합. D-Vlog 공식(SPD+audio 0.808 기준).
SPD(기하 공분산 동역학) + TCN(시간 파형) + audio. 비교: SPD+aud / TCN+aud / SPD+TCN+aud.
5-seed 앙상블. 결과: results/exp87_combine.csv
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); dev='cuda'; T=256; W=64; STR=32; EP=60
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
def resamp(X,t=T):
    n=len(X);idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def win_covs(seq):
    d=seq.shape[1];cs=[]
    for s in range(0,T-W+1,STR):
        seg=seq[s:s+W];seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);cs.append(c+1e-3*np.eye(d))
    return np.array(cs)
class SPDNet(nn.Module):
    def __init__(self,d,out=16):
        super().__init__();self.W=nn.Parameter(torch.randn(out,d)*0.1);self.out=out
        self.pre=nn.Sequential(nn.Linear(out*(out+1)//2,64),nn.ReLU())
    def sl(self,M):
        Bm=self.W@M@self.W.transpose(0,1)+1e-4*torch.eye(self.out,device=M.device)
        ev,U=torch.linalg.eigh(Bm);ev=ev.clamp(min=1e-4);M2=U@torch.diag_embed(ev)@U.transpose(1,2)
        ev2,U2=torch.linalg.eigh(M2+1e-6*torch.eye(self.out,device=M.device))
        L=U2@torch.diag_embed(torch.log(ev2.clamp(min=1e-6)))@U2.transpose(1,2)
        idx=torch.triu_indices(self.out,self.out);return L[:,idx[0],idx[1]]
    def forward(self,covs):
        Bs,K,d,_=covs.shape;v=self.sl(covs.reshape(Bs*K,d,d)).reshape(Bs,K,-1);return self.pre(v).mean(1)
class TCN(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(
            nn.Conv1d(C,48,3,padding=1,dilation=1),nn.BatchNorm1d(48),nn.ReLU(),
            nn.Conv1d(48,48,3,padding=2,dilation=2),nn.BatchNorm1d(48),nn.ReLU(),
            nn.Conv1d(48,96,3,padding=4,dilation=4),nn.BatchNorm1d(96),nn.ReLU(),
            nn.Conv1d(96,96,3,padding=8,dilation=8),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Model(nn.Module):
    def __init__(self,d,Cv,Ca,use_spd,use_tcn):
        super().__init__();self.us=use_spd;self.ut=use_tcn;fin=96  # audio
        if use_spd:self.spd=SPDNet(d);fin+=64
        if use_tcn:self.tcn=TCN(Cv);fin+=96
        self.a=CNNb(Ca)
        self.head=nn.Sequential(nn.Linear(fin,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,xv,xa):
        parts=[self.a(xa)]
        if self.us:parts.append(self.spd(covs))
        if self.ut:parts.append(self.tcn(xv))
        return self.head(torch.cat(parts,1)).squeeze(-1)
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
COVS=[];VIS=[];AUD=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);COVS.append(win_covs(resamp(Zz)));VIS.append(resamp(Zz))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);AUD.append(resamp(Aa))
COVS=np.array(COVS);VIS=np.array(VIS);AUD=np.array(AUD);Ca=AUD.shape[2]
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
def run(us,ut):
    preds=[]
    for seed in range(5):
        torch.manual_seed(seed);net=Model(20,20,Ca,us,ut).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
        pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Ct=T3(COVS[tr]);Vt=T3(VIS[tr]).transpose(1,2);At=T3(AUD[tr]).transpose(1,2);yt=T3(Y[tr])
        Cv=T3(COVS[va]);Vv=T3(VIS[va]).transpose(1,2);Av=T3(AUD[va]).transpose(1,2)
        Ce=T3(COVS[te]);Ve=T3(VIS[te]).transpose(1,2);Ae=T3(AUD[te]).transpose(1,2)
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
        if bp is not None:preds.append(bp)
    ens=np.mean(preds,0);return roc_auc_score(Y[te],ens),f1_score(Y[te],(ens>0.5).astype(int))
out=[]
for name,us,ut in [('SPD+aud',1,0),('TCN+aud',0,1),('SPD+TCN+aud',1,1)]:
    a,f=run(us,ut);print(f'  {name:14s} AUC={a:.3f} F1={f:.3f}',flush=True);out.append([name,a,f])
print(f'\n[비교] exp85 SPD+aud 0.808/0.788',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp87_combine.csv','w') as fo:
    fo.write('model,AUC,F1\n')
    for r in out:fo.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f}\n')
print('DONE → exp87_combine.csv',flush=True)
