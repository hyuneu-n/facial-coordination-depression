"""
exp80 — 공식 D-Vlog split(fold)서 경량 AV+coupling 벤치마크 → 발표 SOTA 직접 비교.
train서 학습, valid서 early-stop, test서 F1/AUC. 발표: D-Vlog 원논문 F1~0.63, 후속들.
경량 모델이 대등/우위면 "경량+해석 필적" 기여. 결과: results/exp80_official.csv
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); dev='cuda'; T=256; EP=60
def resamp(X,t=T):
    n=len(X)
    if n==t:return X
    idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Net(nn.Module):
    def __init__(self,Cv,Ca,tan):
        super().__init__();self.v=CNNb(Cv);self.a=CNNb(Ca)
        self.g=nn.Sequential(nn.Linear(tan,96),nn.ReLU(),nn.Dropout(0.4),nn.Linear(96,48))
        self.head=nn.Sequential(nn.Linear(96+96+48,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,xv,xa,g):
        return self.head(torch.cat([self.v(xv),self.a(xa),self.g(g)],1)).squeeze(-1)
# load with fold
rows=list(csv.DictReader(open(R/'labels.csv')))
print('fold 컬럼값:', set(r.get('fold','?') for r in rows[:50]),flush=True)
data={'train':[],'valid':[],'test':[]}
vraw=[];araw=[];Y=[];F=[]
for r in rows:
    idx=r['index'].strip();fold=r.get('fold','').strip().lower()
    fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
    if not(fv.exists() and fa.exists()):continue
    if fold not in('train','valid','test'):
        if 'val' in fold: fold='valid'
        elif fold=='':continue
    try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
    except:continue
    if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136 or Au.ndim!=2 or len(Au)<10:continue
    vraw.append(nf(V));araw.append(Au);Y.append(1 if r['label'].strip().lower().startswith('depress') else 0);F.append(fold)
print('총',len(Y),'fold분포',{k:F.count(k) for k in set(F)},flush=True)
pca=PCA(20).fit(np.vstack([vraw[i][::3] for i in range(len(vraw)) if F[i]=='train']))
VIS=[];AUD=[];COV=[]
for i in range(len(Y)):
    Z=pca.transform(vraw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);VIS.append(resamp(Zz))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);AUD.append(resamp(Aa))
    c,_=ledoit_wolf(Z);COV.append(c+1e-3*np.eye(20))
VIS=np.array(VIS);AUD=np.array(AUD);COV=np.array(COV);Y=np.array(Y);F=np.array(F)
tr=F=='train';va=F=='valid';te=F=='test'
Ts=TangentSpace(metric='riemann').fit(COV[tr]);Zt=Ts.transform(COV);sc=StandardScaler().fit(Zt[tr]);G=sc.transform(Zt)
Ca=AUD.shape[2]
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
res=[]
for seed in range(5):
    torch.manual_seed(seed);net=Net(20,Ca,G.shape[1]).to(dev)
    opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xv=T3(VIS[tr]).transpose(1,2);Xa=T3(AUD[tr]).transpose(1,2);Gg=T3(G[tr]);yt=T3(Y[tr])
    Xvv=T3(VIS[va]).transpose(1,2);Xav=T3(AUD[va]).transpose(1,2);Gv=T3(G[va])
    Xvt=T3(VIS[te]).transpose(1,2);Xat=T3(AUD[te]).transpose(1,2);Gt=T3(G[te])
    n=tr.sum();bs=64;bestva=0;bestp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Xv[b],Xa[b],Gg[b]),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            pv=torch.sigmoid(net(Xvv,Xav,Gv)).cpu().numpy();pt=torch.sigmoid(net(Xvt,Xat,Gt)).cpu().numpy()
        try:
            av=roc_auc_score(Y[va],pv)
            if av>bestva:bestva=av;bestp=pt
        except:pass
    if bestp is None:continue
    pred=(bestp>0.5).astype(int)
    res.append((roc_auc_score(Y[te],bestp),f1_score(Y[te],pred),precision_score(Y[te],pred),recall_score(Y[te],pred)))
res=np.array(res)
print(f'\n===== D-Vlog 공식 test (경량 AV+coupling, 5seed) =====',flush=True)
print(f'  AUC={res[:,0].mean():.3f}±{res[:,0].std():.3f}',flush=True)
print(f'  F1 ={res[:,1].mean():.3f}±{res[:,1].std():.3f}  P={res[:,2].mean():.3f} R={res[:,3].mean():.3f}',flush=True)
print(f'  [비교] D-Vlog 원논문(Yoon2022) F1~0.63, 후속 멀티모달 F1 0.65~0.70대',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp80_official.csv','w') as f:
    f.write('metric,mean,std\n');f.write(f'AUC,{res[:,0].mean():.4f},{res[:,0].std():.4f}\nF1,{res[:,1].mean():.4f},{res[:,1].std():.4f}\n')
print('DONE → exp80_official.csv',flush=True)
