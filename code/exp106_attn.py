"""exp106 — 시간 해석: 모델이 세션 시간축 어디를 보나(attention). 조기탐지와 교차검증.
SPD+GRU+temporal attention. 윈도우별 attention 가중치를 MDD/HC 그룹평균해 세션위치별 시각화.
앞부분에 집중하면 조기탐지(앞40% 충분)와 자기일관.  fig9.
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); dev='cuda'; T=256; W=64; STR=32; EP=60; NSEED=6; d=20
K=len(range(0,T-W+1,STR))
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
        Bs,Kk,dd,_=covs.shape;v=self.sl(covs.reshape(Bs*Kk,dd,dd)).reshape(Bs,Kk,-1);return self.pre(v)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Model(nn.Module):
    def __init__(self,Ca):
        super().__init__();self.spd=SPDseq(d);self.gru=nn.GRU(64,32,batch_first=True,bidirectional=True)
        self.att=nn.Linear(64,1);self.a=CNNb(Ca);self.cls=nn.Sequential(nn.Linear(64+96,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,xa,ret_attn=False):
        o,_=self.gru(self.spd(covs));w=torch.softmax(self.att(o),1)  # (B,K,1)
        pooled=(w*o).sum(1)
        out=self.cls(torch.cat([pooled,self.a(xa)],1)).squeeze(-1)
        return (out,w.squeeze(-1)) if ret_attn else out
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
COVS=[];AUD=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);COVS.append(win_covs(resamp(Zz)))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);AUD.append(resamp(Aa))
COVS=np.array(COVS);AUD=np.array(AUD);Ca=AUD.shape[2]
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
attn_all=[];aucs=[]
for seed in range(NSEED):
    torch.manual_seed(seed);net=Model(Ca).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Ct=T3(COVS[tr]);At=T3(AUD[tr]).transpose(1,2);yt=T3(Y[tr]);Cv=T3(COVS[va]);Av=T3(AUD[va]).transpose(1,2)
    n=tr.sum();bs=64;bva=0;bstate=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Ct[b],At[b]),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():a=roc_auc_score(Y[va],torch.sigmoid(net(Cv,Av)).cpu().numpy())
        if a>bva:bva=a;bstate={k:v.detach().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(bstate);net.eval()
    Ce=T3(COVS[te]);Ae=T3(AUD[te]).transpose(1,2)
    with torch.no_grad():o,w=net(Ce,Ae,ret_attn=True)
    aucs.append(roc_auc_score(Y[te],torch.sigmoid(o).cpu().numpy()))
    attn_all.append(w.cpu().numpy())  # (Nte,K)
attn=np.mean(attn_all,0)  # seed평균 (Nte,K)
Yte=Y[te]
aMDD=attn[Yte==1].mean(0);aHC=attn[Yte==0].mean(0);aAll=attn.mean(0)
pos=[(32*k+32)/T*100 for k in range(K)]  # 윈도우 세션위치 %
print(f'평균 test AUC={np.mean(aucs):.4f}',flush=True)
print('세션위치(%) : '+' '.join(f'{p:.0f}' for p in pos),flush=True)
print('attention전체: '+' '.join(f'{a:.3f}' for a in aAll),flush=True)
front=sum(aAll[k] for k in range(K) if pos[k]<=40);print(f'앞40% 이내 attention 합={front:.3f} (균등이면 {sum(1 for p in pos if p<=40)/K:.3f})',flush=True)
# fig9
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os
for c in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c):fm.fontManager.addfont(c);plt.rcParams['font.family']=fm.FontProperties(fname=c).get_name()
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(6.4,4.4))
plt.plot(pos,aAll,'-o',color='#1565C0',lw=2.4,label='전체',zorder=3)
plt.plot(pos,aMDD,'--o',color='#c0392b',lw=1.8,label='우울군',alpha=0.8)
plt.plot(pos,aHC,'--o',color='#2E7D32',lw=1.8,label='건강군',alpha=0.8)
plt.axhline(1.0/K,ls=':',color='#9E9E9E',lw=1);plt.text(pos[-1],1.0/K+0.003,'균등(no focus)',color='#9E9E9E',fontsize=9,ha='right')
plt.xlabel('세션 위치 (%)');plt.ylabel('attention 가중치')
plt.title('모델이 세션 어디를 보나 — 시간 attention\n(조기탐지와 교차검증, D-Vlog)',fontsize=12)
plt.legend(fontsize=9);plt.tight_layout()
plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig9_attn.png',dpi=145,bbox_inches='tight')
print('DONE fig9_attn.png',flush=True)
