"""exp102 — multi-scale coordination. 순간(W32)+중간(W64)+지속(W128) 협응을 SPD로 각각 학습 후 융합.
SPDNet·GRU 스케일간 공유(과적합 억제). baseline=단일 W64. 이기면 기하 novelty. D-Vlog 공식 8seed.
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); dev='cuda'; T=256; EP=60; NSEED=8; d=20
SCALES=[(32,16),(64,32),(128,64)]  # (W,STR): 순간/중간/지속
def nf(V):
    T2=V.shape[0];Pp=V.reshape(T2,68,2).astype(float);Pp=Pp-Pp.mean(1,keepdims=True)
    sc=np.sqrt((Pp**2).sum(2).mean(1,keepdims=True))+1e-6;return (Pp/sc[:,None]).reshape(T2,136)
def resamp(X,t=T):
    n=len(X);idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def win_covs(seq,Wn,St):
    cs=[]
    for s in range(0,T-Wn+1,St):
        seg=seq[s:s+Wn];seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);cs.append(c+1e-3*np.eye(d))
    return np.array(cs)
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
    def __init__(self,Ca,multi):
        super().__init__();self.spd=SPDseq(d);self.multi=multi
        self.gru=nn.GRU(64,32,batch_first=True,bidirectional=True);self.a=CNNb(Ca)
        nsc=len(SCALES) if multi else 1
        self.fuse=nn.Sequential(nn.Linear(64*nsc,64),nn.ReLU()) if multi else nn.Identity()
        self.cls=nn.Sequential(nn.Linear(64+96,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def enc(self,covs):o,_=self.gru(self.spd(covs));return o.mean(1)  # 공유 SPD+GRU
    def forward(self,covs_list,xa):
        if self.multi:
            embs=[self.enc(c) for c in covs_list];h=self.fuse(torch.cat(embs,1))
        else:
            h=self.enc(covs_list[0])
        return self.cls(torch.cat([h,self.a(xa)],1)).squeeze(-1)
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
# 스케일별 cov 미리계산
SC={si:[] for si in range(len(SCALES))};AUD=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);Zr=resamp(Zz)
    for si,(Wn,St) in enumerate(SCALES):SC[si].append(win_covs(Zr,Wn,St))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);AUD.append(resamp(Aa))
for si in SC:SC[si]=np.array(SC[si])
AUD=np.array(AUD);Ca=AUD.shape[2]
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
def run(multi,seed):
    torch.manual_seed(seed);net=Model(Ca,multi).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    def mk(mask):
        cl=[T3(SC[si][mask]) for si in range(len(SCALES))] if multi else [T3(SC[1][mask])]
        return cl,T3(AUD[mask]).transpose(1,2)
    Ctr,Atr=mk(tr);Cva,Ava=mk(va);Cte,Ate=mk(te);yt=T3(Y[tr])
    n=tr.sum();bs=64;bva=0;bp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad()
            cl=[c[b] for c in Ctr];loss=lf(net(cl,Atr[b]),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():pv=torch.sigmoid(net(Cva,Ava)).cpu().numpy();pt=torch.sigmoid(net(Cte,Ate)).cpu().numpy()
        try:
            a=roc_auc_score(Y[va],pv)
            if a>bva:bva=a;bp=pt
        except:pass
    return roc_auc_score(Y[te],bp),f1_score(Y[te],(bp>0.5).astype(int))
print(f'=== 단일스케일(W64) vs multi-scale({len(SCALES)}) D-Vlog 공식 {NSEED}seed ===',flush=True)
b=[];m=[];win=0
for s in range(NSEED):
    ab,fb=run(False,s);am,fm=run(True,s);b.append(ab);m.append(am);win+=int(am>ab)
    print(f'  seed{s}: single={ab:.4f} multi={am:.4f} {"multi승" if am>ab else "single승"}',flush=True)
print(f'  >> single {np.mean(b):.4f}±{np.std(b):.4f} | multi-scale {np.mean(m):.4f}±{np.std(m):.4f} | multi승률 {win}/{NSEED}',flush=True)
print('DONE',flush=True)
