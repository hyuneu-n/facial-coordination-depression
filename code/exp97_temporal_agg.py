"""exp97 — 시간 집계 방식 ablation (D-Vlog 공식 split). SPD 윈도우 임베딩을 어떻게 모으나.
mean(현재/베이스) vs meanstd vs attention vs GRU vs TCN. 시간모델이 mean 이기면 '시간 협응 모델링' 방법 기여.
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
class SPDseq(nn.Module):  # 윈도우별 임베딩 시퀀스 (B,K,64) 반환
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
        Bs,K,d,_=covs.shape;v=self.sl(covs.reshape(Bs*K,d,d)).reshape(Bs,K,-1);return self.pre(v)  # (B,K,64)
class Agg(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='meanstd':self.fc=nn.Linear(128,64)
        elif kind=='attn':self.att=nn.Linear(64,1)
        elif kind=='gru':self.gru=nn.GRU(64,32,batch_first=True,bidirectional=True)
        elif kind=='tcn':self.c1=nn.Conv1d(64,64,3,padding=1);self.c2=nn.Conv1d(64,64,3,padding=2,dilation=2)
    def forward(self,x):  # x:(B,K,64)
        if self.kind=='mean':return x.mean(1)
        if self.kind=='meanstd':return torch.relu(self.fc(torch.cat([x.mean(1),x.std(1)],1)))
        if self.kind=='attn':w=torch.softmax(self.att(x),1);return (w*x).sum(1)
        if self.kind=='gru':o,_=self.gru(x);return o.mean(1)
        if self.kind=='tcn':h=torch.relu(self.c1(x.transpose(1,2)));h=torch.relu(self.c2(h));return h.mean(2)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Model(nn.Module):
    def __init__(self,d,Ca,kind):
        super().__init__();self.spd=SPDseq(d);self.agg=Agg(kind);self.a=CNNb(Ca)
        self.cls=nn.Sequential(nn.Linear(64+96,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,xa):return self.cls(torch.cat([self.agg(self.spd(covs)),self.a(xa)],1)).squeeze(-1)
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
COVS=[];AUD=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);COVS.append(win_covs(resamp(Zz)))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);AUD.append(resamp(Aa))
COVS=np.array(COVS);AUD=np.array(AUD);Ca=AUD.shape[2]
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
def run(kind):
    preds=[]
    for seed in range(5):
        torch.manual_seed(seed);net=Model(20,Ca,kind).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
        pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Ct=T3(COVS[tr]);At=T3(AUD[tr]).transpose(1,2);yt=T3(Y[tr])
        Cv=T3(COVS[va]);Av=T3(AUD[va]).transpose(1,2);Ce=T3(COVS[te]);Ae=T3(AUD[te]).transpose(1,2)
        n=tr.sum();bs=64;bva=0;bp=None
        for ep in range(EP):
            net.train();perm=torch.randperm(n)
            for i in range(0,n,bs):
                b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Ct[b],At[b]),yt[b]);loss.backward();opt.step()
            net.eval()
            with torch.no_grad():pv=torch.sigmoid(net(Cv,Av)).cpu().numpy();pt=torch.sigmoid(net(Ce,Ae)).cpu().numpy()
            try:
                a=roc_auc_score(Y[va],pv)
                if a>bva:bva=a;bp=pt
            except:pass
        if bp is not None:preds.append(bp)
    ens=np.mean(preds,0);return roc_auc_score(Y[te],ens),f1_score(Y[te],(ens>0.5).astype(int))
out=[]
for k in ['mean','meanstd','attn','gru','tcn']:
    a,f=run(k);print(f'  {k:9s} AUC={a:.4f} F1={f:.4f}',flush=True);out.append([k,a,f])
print('\n[베이스=mean 0.807]. 시간집계가 mean 넘으면 방법 기여',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp97_tagg.csv','w') as fo:
    fo.write('agg,AUC,F1\n')
    for r in out:fo.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f}\n')
print('DONE',flush=True)
