"""exp101 — 저차원 변형. RiemVel 210차원(실패) 대신 '변화량 크기 스칼라 1개'만 GRU에 추가.
또 velocity-gated attention pooling도 비교. baseline=GRU. D-Vlog 공식 8seed. 이기면 경량 algorithmic novelty.
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
def vel_mag(covs):  # (K,)  곡면 측지속도 크기(스칼라)
    out=[0.0]
    for t in range(len(covs)-1):
        ish=inv_sqrt(covs[t]);Tt=ish@covs[t+1]@ish;out.append(float(np.linalg.norm(logm_vec(Tt))))
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
    def __init__(self,Ca,mode):
        super().__init__();self.spd=SPDseq(d);self.mode=mode  # base / scalar / velattn
        gin=65 if mode=='scalar' else 64
        self.gru=nn.GRU(gin,32,batch_first=True,bidirectional=True);self.a=CNNb(Ca)
        if mode=='velattn':self.att=nn.Linear(64+1,1)
        self.cls=nn.Sequential(nn.Linear(64+96,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,vmag,xa):
        s=self.spd(covs)  # (B,K,64)
        if self.mode=='scalar':
            o,_=self.gru(torch.cat([s,vmag.unsqueeze(-1)],2));pooled=o.mean(1)
        elif self.mode=='velattn':
            o,_=self.gru(s);w=torch.softmax(self.att(torch.cat([o,vmag.unsqueeze(-1)],2)),1);pooled=(w*o).sum(1)
        else:
            o,_=self.gru(s);pooled=o.mean(1)
        return self.cls(torch.cat([pooled,self.a(xa)],1)).squeeze(-1)
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
COVS=[];VM=[];AUD=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);cv=win_covs(resamp(Zz))
    COVS.append(cv);VM.append(vel_mag(cv))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);AUD.append(resamp(Aa))
COVS=np.array(COVS);VM=np.array(VM);AUD=np.array(AUD);Ca=AUD.shape[2]
mm=VM[Fd=='train'].mean();ss=VM[Fd=='train'].std()+1e-6;VM=(VM-mm)/ss
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
def run(mode,seed):
    torch.manual_seed(seed);net=Model(Ca,mode).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Ct=T3(COVS[tr]);Mt=T3(VM[tr]);At=T3(AUD[tr]).transpose(1,2);yt=T3(Y[tr])
    Cv=T3(COVS[va]);Mv=T3(VM[va]);Av=T3(AUD[va]).transpose(1,2)
    Ce=T3(COVS[te]);Me=T3(VM[te]);Ae=T3(AUD[te]).transpose(1,2)
    n=tr.sum();bs=64;bva=0;bp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Ct[b],Mt[b],At[b]),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():pv=torch.sigmoid(net(Cv,Mv,Av)).cpu().numpy();pt=torch.sigmoid(net(Ce,Me,Ae)).cpu().numpy()
        try:
            a=roc_auc_score(Y[va],pv)
            if a>bva:bva=a;bp=pt
        except:pass
    return roc_auc_score(Y[te],bp)
res={m:[] for m in ['base','scalar','velattn']}
for s in range(NSEED):
    line=f'seed{s}: '
    for m in ['base','scalar','velattn']:
        a=run(m,s);res[m].append(a);line+=f'{m}={a:.4f} '
    print('  '+line,flush=True)
print('\n=== 요약 (D-Vlog 공식) ===',flush=True)
for m in ['base','scalar','velattn']:
    w=sum(res[m][s]>res['base'][s] for s in range(NSEED))
    print(f'  {m:8s} {np.mean(res[m]):.4f}±{np.std(res[m]):.4f}'+(f' | base대비 승 {w}/{NSEED}' if m!='base' else ''),flush=True)
print('DONE',flush=True)
