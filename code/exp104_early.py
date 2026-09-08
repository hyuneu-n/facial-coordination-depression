"""exp104 — early-detection 곡선 (proto-전조). 세션 앞 X%만 보고 우울 탐지되나?
SPD+GRU+audio를 full로 학습, 앞 20/40/60/80/100%로 테스트. 초반에 잡히면 전조방향 실증.
결과: results/exp104_early.csv + figs/fig7_early.png
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); dev='cuda'; T=256; W=64; STR=32; EP=60; NSEED=5; d=20
FRACS=[0.2,0.4,0.6,0.8,1.0]
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
        Bs,K,dd,_=covs.shape;v=self.sl(covs.reshape(Bs*K,dd,dd)).reshape(Bs,K,-1);return self.pre(v)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Model(nn.Module):
    def __init__(self,Ca):
        super().__init__();self.spd=SPDseq(d);self.gru=nn.GRU(64,32,batch_first=True,bidirectional=True)
        self.a=CNNb(Ca);self.cls=nn.Sequential(nn.Linear(64+96,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,xa):o,_=self.gru(self.spd(covs));return self.cls(torch.cat([o.mean(1),self.a(xa)],1)).squeeze(-1)
rows=list(csv.DictReader(open(R/'labels.csv')))
rawV=[];rawA=[];Y=[];Fd=[]
for r in rows:
    idx=r['index'].strip();fold=r.get('fold','').strip().lower()
    if 'val' in fold:fold='valid'
    if fold not in('train','valid','test'):continue
    fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
    if not(fv.exists() and fa.exists()):continue
    try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
    except:continue
    if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136 or Au.ndim!=2 or len(Au)<10:continue
    rawV.append(nf(V));rawA.append(Au);Y.append(1 if r['label'].strip().lower().startswith('depress') else 0);Fd.append(fold)
Y=np.array(Y);Fd=np.array(Fd)
pca=PCA(d).fit(np.vstack([rawV[i][::3] for i in range(len(rawV)) if Fd[i]=='train']))
# 앞 frac%만 잘라 resample → cov/audio. frac=1.0 = 전체(학습용)
def build(frac):
    COVS=[];AUD=[]
    for i in range(len(Y)):
        Vv=rawV[i];Aa=rawA[i]
        nv=max(W+2,int(len(Vv)*frac));na=max(4,int(len(Aa)*frac))
        Z=pca.transform(Vv[:nv]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);COVS.append(win_covs(resamp(Zz)))
        Ax=Aa[:na];Ax=(Ax-Ax.mean(0))/(Ax.std(0)+1e-6);AUD.append(resamp(Ax))
    return np.array(COVS),np.array(AUD)
FULL_C,FULL_A=build(1.0);Ca=FULL_A.shape[2]
PREF={f:build(f) for f in FRACS if f<1.0};PREF[1.0]=(FULL_C,FULL_A)
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
res={f:[] for f in FRACS}
for seed in range(NSEED):
    torch.manual_seed(seed);net=Model(Ca).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Ct=T3(FULL_C[tr]);At=T3(FULL_A[tr]).transpose(1,2);yt=T3(Y[tr])
    Cv=T3(FULL_C[va]);Av=T3(FULL_A[va]).transpose(1,2)
    n=tr.sum();bs=64;bva=0;bstate=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Ct[b],At[b]),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():a=roc_auc_score(Y[va],torch.sigmoid(net(Cv,Av)).cpu().numpy())
        if a>bva:bva=a;bstate={k:v.detach().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(bstate);net.eval()
    for f in FRACS:
        C,A=PREF[f];Ce=T3(C[te]);Ae=T3(A[te]).transpose(1,2)
        with torch.no_grad():pt=torch.sigmoid(net(Ce,Ae)).cpu().numpy()
        res[f].append(roc_auc_score(Y[te],pt))
print('=== early-detection: 세션 앞 X%만 보고 테스트 AUC (D-Vlog 공식) ===',flush=True)
means=[];stds=[]
for f in FRACS:
    m=np.mean(res[f]);s=np.std(res[f]);means.append(m);stds.append(s)
    print(f'  앞 {int(f*100):3d}%: AUC={m:.4f} ± {s:.4f}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp104_early.csv','w') as fo:
    fo.write('frac,AUC,std\n')
    for f,m,s in zip(FRACS,means,stds):fo.write(f'{f},{m:.4f},{s:.4f}\n')
# figure
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os
for c in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c):fm.fontManager.addfont(c);plt.rcParams['font.family']=fm.FontProperties(fname=c).get_name()
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(6.2,4.4))
x=[f*100 for f in FRACS]
plt.errorbar(x,means,yerr=stds,marker='o',color='#1565C0',lw=2.2,capsize=4)
plt.axhline(means[-1],ls='--',color='#9E9E9E',lw=1);plt.axhline(0.5,ls=':',color='#c0392b',lw=1)
plt.text(22,0.505,'chance',color='#c0392b',fontsize=9)
for xi,m in zip(x,means):plt.text(xi,m+0.008,f'{m:.2f}',ha='center',fontsize=10,fontweight='bold')
plt.xlabel('관찰한 세션 앞부분 (%)');plt.ylabel('우울 탐지 AUC')
plt.title('조기 탐지 곡선 — 세션 앞부분만으로 우울이 잡히나\n(전조 방향의 실증, D-Vlog)',fontsize=12)
plt.ylim(0.5,0.9);plt.tight_layout()
plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig7_early.png',dpi=145,bbox_inches='tight')
print('DONE fig7_early.png',flush=True)
