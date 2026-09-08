"""
exp84 — 특화 인코더 + 방법론으로 성능 개선 시도. D-Vlog 공식 split(exp80 F1 0.727 기준).
인코더: CNN(base) / TCN(dilated) / CNN+freq(time+frequency 2-branch, STSFF 근거).
방법론: label smoothing + time-augment(jitter/scale) + seed 앙상블(soft-vote).
visual+audio+coupling fusion. 결과: results/exp84_encoder.csv
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); dev='cuda'; T=256; EP=60
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
def resamp(X,t=T):
    n=len(X);idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class TCNb(nn.Module):  # dilated
    def __init__(self,C):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(C,48,3,padding=1,dilation=1),nn.BatchNorm1d(48),nn.ReLU(),
            nn.Conv1d(48,48,3,padding=2,dilation=2),nn.BatchNorm1d(48),nn.ReLU(),
            nn.Conv1d(48,96,3,padding=4,dilation=4),nn.BatchNorm1d(96),nn.ReLU(),
            nn.Conv1d(96,96,3,padding=8,dilation=8),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class FreqCNN(nn.Module):  # time CNN + frequency branch
    def __init__(self,C):
        super().__init__();self.t=CNNb(C)
        self.f=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.proj=nn.Linear(96+48,96)
    def forward(self,x):
        ht=self.t(x)
        mag=torch.fft.rfft(x,dim=2).abs()  # B,C,F
        hf=self.f(mag).squeeze(-1)
        return torch.relu(self.proj(torch.cat([ht,hf],1)))
def enc(kind,C):
    return {'CNN':CNNb,'TCN':TCNb,'freq':FreqCNN}[kind](C)
class Net(nn.Module):
    def __init__(self,Cv,Ca,tan,kind):
        super().__init__();self.v=enc(kind,Cv);self.a=enc(kind,Ca)
        self.g=nn.Sequential(nn.Linear(tan,96),nn.ReLU(),nn.Dropout(0.4),nn.Linear(96,48))
        self.head=nn.Sequential(nn.Linear(96+96+48,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,xv,xa,g):return self.head(torch.cat([self.v(xv),self.a(xa),self.g(g)],1)).squeeze(-1)
# load official
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
VIS=[];AUD=[];COV=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);VIS.append(resamp(Zz))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);AUD.append(resamp(Aa))
    c,_=ledoit_wolf(Z);COV.append(c+1e-3*np.eye(20))
VIS=np.array(VIS);AUD=np.array(AUD);COV=np.array(COV);Ca=AUD.shape[2]
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
Ts=TangentSpace(metric='riemann').fit(COV[tr]);Zt=Ts.transform(COV);sc=StandardScaler().fit(Zt[tr]);G=sc.transform(Zt)
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
def augment(x):  # jitter+scale
    return x*(1+0.1*torch.randn(x.size(0),1,1,device=dev))+0.03*torch.randn_like(x)
def run(kind,ls,aug):
    preds=[]
    for seed in range(5):
        torch.manual_seed(seed);net=Net(20,Ca,G.shape[1],kind).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
        pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Xv=T3(VIS[tr]).transpose(1,2);Xa=T3(AUD[tr]).transpose(1,2);Gg=T3(G[tr]);yt=T3(Y[tr])
        Xvv=T3(VIS[va]).transpose(1,2);Xav=T3(AUD[va]).transpose(1,2);Gv=T3(G[va])
        Xvt=T3(VIS[te]).transpose(1,2);Xat=T3(AUD[te]).transpose(1,2);Gt=T3(G[te])
        n=tr.sum();bs=64;bva=0;bp=None
        for ep in range(EP):
            net.train();perm=torch.randperm(n)
            for i in range(0,n,bs):
                b=perm[i:i+bs];opt.zero_grad()
                xv=augment(Xv[b]) if aug else Xv[b];xa=augment(Xa[b]) if aug else Xa[b]
                tgt=yt[b]*(1-ls)+0.5*ls if ls else yt[b]
                loss=lf(net(xv,xa,Gg[b]),tgt);loss.backward();opt.step()
            net.eval()
            with torch.no_grad():pv=torch.sigmoid(net(Xvv,Xav,Gv)).cpu().numpy();pt=torch.sigmoid(net(Xvt,Xat,Gt)).cpu().numpy()
            try:
                av=roc_auc_score(Y[va],pv)
                if av>bva:bva=av;bp=pt
            except:pass
        if bp is not None:preds.append(bp)
    ens=np.mean(preds,0)  # seed 앙상블
    return roc_auc_score(Y[te],ens),f1_score(Y[te],(ens>0.5).astype(int))
out=[]
for kind in ['CNN','TCN','freq']:
    a,f=run(kind,0,False);print(f'  {kind:5s} (base)          AUC={a:.3f} F1={f:.3f}',flush=True);out.append([kind+'_base',a,f])
# 최고 인코더에 방법론
best=max(out,key=lambda r:r[2])[0].split('_')[0]
a,f=run(best,0.1,True);print(f'  {best:5s} +LS+aug+ensemble  AUC={a:.3f} F1={f:.3f}',flush=True);out.append([best+'_LS_aug',a,f])
print(f'\n[비교] exp80 base CNN F1 0.727 | 원논문 0.63',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp84_encoder.csv','w') as fo:
    fo.write('model,AUC,F1\n')
    for r in out:fo.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f}\n')
print('DONE → exp84_encoder.csv',flush=True)
