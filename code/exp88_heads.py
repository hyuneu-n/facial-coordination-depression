"""
exp88 — 다른 fusion 헤드 비교 (인코더 SPD+audio 고정). D-Vlog 공식(SPD+aud concat 0.80 기준).
헤드: concat / gated / cross-attention / bilinear(low-rank). 5-seed 앙상블.
결과: results/exp88_heads.csv
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
        I=torch.eye(self.out,device=M.device,dtype=torch.float64)
        Bm=(self.W.double()@M.double()@self.W.double().transpose(0,1))+1e-2*I
        ev,U=torch.linalg.eigh(Bm);ev=ev.clamp(min=1e-3);M2=U@torch.diag_embed(ev)@U.transpose(1,2)
        ev2,U2=torch.linalg.eigh(M2+1e-4*I)
        L=U2@torch.diag_embed(torch.log(ev2.clamp(min=1e-4)))@U2.transpose(1,2)
        idx=torch.triu_indices(self.out,self.out);return L[:,idx[0],idx[1]].float()
    def forward(self,covs):
        Bs,K,d,_=covs.shape;v=self.sl(covs.reshape(Bs*K,d,d)).reshape(Bs,K,-1);return self.pre(v).mean(1)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Model(nn.Module):
    def __init__(self,d,Ca,head):
        super().__init__();self.spd=SPDNet(d);self.a=CNNb(Ca);self.head_type=head
        self.ps=nn.Linear(64,64);self.pa=nn.Linear(96,64)  # 공통 차원
        if head=='concat':self.cls=nn.Sequential(nn.Linear(64+96,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
        elif head=='gated':self.gate=nn.Linear(128,2);self.cls=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
        elif head=='xattn':self.mha=nn.MultiheadAttention(64,4,batch_first=True);self.cls=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
        elif head=='bilinear':self.cls=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,xa):
        s=self.spd(covs);a=self.a(xa)
        if self.head_type=='concat':return self.cls(torch.cat([s,a],1)).squeeze(-1)
        sp=torch.relu(self.ps(s));ap=torch.relu(self.pa(a))
        if self.head_type=='gated':
            g=torch.softmax(self.gate(torch.cat([sp,ap],1)),1);f=g[:,0:1]*sp+g[:,1:2]*ap;return self.cls(f).squeeze(-1)
        if self.head_type=='xattn':
            tok=torch.stack([sp,ap],1);o,_=self.mha(tok,tok,tok);return self.cls(o.mean(1)).squeeze(-1)
        if self.head_type=='bilinear':
            f=sp*ap;return self.cls(f).squeeze(-1)
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
def run(head):
    preds=[]
    for seed in range(5):
        torch.manual_seed(seed);net=Model(20,Ca,head).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
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
for h in ['concat','gated','xattn','bilinear']:
    a,f=run(h);print(f'  {h:9s} AUC={a:.3f} F1={f:.3f}',flush=True);out.append([h,a,f])
print(f'\n[비교] concat = exp85/87 SPD+aud ~0.80/0.80',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp88_heads.csv','w') as fo:
    fo.write('head,AUC,F1\n')
    for r in out:fo.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f}\n')
print('DONE → exp88_heads.csv',flush=True)
