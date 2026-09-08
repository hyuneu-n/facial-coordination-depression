"""
exp79 — 경량 coordination-aware AV fusion (멀티모달 강화, 헤드룸).
시각-CNN + 음성-CNN + coupling(기하) + cross-modal 협응(face-voice sync, exp74). generic 아님.
비교: visual / AV / full(AV+coupling+coord). D-Vlog(audio 강)+LMVD. 3seed×5fold. 경량 유지.
결과: results/exp79_av.csv
"""
import numpy as np, warnings, csv, glob
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); dev='cuda'; T=256; EP=40
def resamp(X,t=T):
    n=len(X)
    if n==t:return X
    idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def env_resamp(x,t=T):
    x=np.asarray(x,float)
    if len(x)<3:return np.zeros(t)
    return np.interp(np.linspace(0,len(x)-1,t),np.arange(len(x)),x)
def coord_feats(fe,ve):
    fe=env_resamp(fe);ve=env_resamp(ve);fe=(fe-fe.mean())/(fe.std()+1e-6);ve=(ve-ve.mean())/(ve.std()+1e-6)
    r0=np.corrcoef(fe,ve)[0,1];best=r0
    for L in range(-30,31):
        a,b=(fe[:L],ve[-L:]) if L<0 else ((fe[L:],ve[:-L]) if L>0 else (fe,ve))
        if len(a)>10:
            c=np.corrcoef(a,b)[0,1]
            if abs(c)>abs(best):best=c
    W=T//6;wc=[np.corrcoef(fe[i:i+W],ve[i:i+W])[0,1] for i in range(0,T-W,W)];wc=[c for c in wc if not np.isnan(c)]
    return np.nan_to_num([r0,best,np.mean(wc) if wc else 0,np.std(wc) if wc else 0])
class CNNb(nn.Module):  # 경량 2-conv
    def __init__(self,C):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
                               nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class AVNet(nn.Module):
    def __init__(self,Cv,Ca,tan,mode):
        super().__init__();self.mode=mode
        self.v=CNNb(Cv); d=96
        if mode!='visual': self.a=CNNb(Ca);d+=96
        if mode=='full':
            self.g=nn.Sequential(nn.Linear(tan,96),nn.ReLU(),nn.Dropout(0.4),nn.Linear(96,48));d+=48+4
        self.head=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(0.4),nn.Linear(64,1))
    def forward(self,xv,xa,tan,coord):
        parts=[self.v(xv)]
        if self.mode!='visual':parts.append(self.a(xa))
        if self.mode=='full':parts+=[self.g(tan),coord]
        return self.head(torch.cat(parts,1)).squeeze(-1)
def train_eval(data_tr,data_te,ytr,yte,Cv,Ca,tan,mode,seed):
    torch.manual_seed(seed);net=AVNet(Cv,Ca,tan,mode).to(dev)
    opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=torch.tensor([(ytr==0).sum()/max(1,(ytr==1).sum())],dtype=torch.float32,device=dev)
    lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
    Xv,Xa,Tn,Co=[T3(d) for d in data_tr]; Xv=Xv.transpose(1,2);Xa=Xa.transpose(1,2)
    Xv2,Xa2,Tn2,Co2=[T3(d) for d in data_te];Xv2=Xv2.transpose(1,2);Xa2=Xa2.transpose(1,2)
    yt=T3(ytr);n=len(ytr);bs=64;best=0.5
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad()
            loss=lf(net(Xv[b],Xa[b],Tn[b],Co[b]),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():p=torch.sigmoid(net(Xv2,Xa2,Tn2,Co2)).cpu().numpy()
        try:best=max(best,roc_auc_score(yte,p))
        except:pass
    return best
def run(vis,aud,cov,coord,y,Cv,Ca,name):
    y=np.array(y);cov=np.array(cov);coord=np.array(coord)
    print(f'\n===== {name} (n={len(y)}) =====',flush=True)
    res={'visual':[],'AV':[],'full':[]}
    for seed in range(3):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(vis,y):
            Ts=TangentSpace(metric='riemann').fit(cov[tr]);Ztr=Ts.transform(cov);sc=StandardScaler().fit(Ztr[tr]);Z=sc.transform(Ztr)
            dtr=(vis[tr],aud[tr],Z[tr],coord[tr]);dte=(vis[te],aud[te],Z[te],coord[te])
            for m in res:
                res[m].append(train_eval(dtr,dte,y[tr],y[te],Cv,Ca,Z.shape[1],m,seed))
        print(f'  seed{seed} done',flush=True)
    for m in res:print(f'  {m:8s} AUC={np.mean(res[m]):.3f}±{np.std(res[m]):.3f}',flush=True)
    return [name,len(y),np.mean(res['visual']),np.mean(res['AV']),np.mean(res['full'])]
out=[]
# ---- D-Vlog ----
print('D-Vlog 로딩...',flush=True)
R=B/'D-Vlog'
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
vraw=[];araw=[];fenv=[];venv=[];y=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
    if not(fv.exists() and fa.exists()):continue
    try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
    except:continue
    if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136 or Au.ndim!=2 or len(Au)<10:continue
    Vn=nf(V);vraw.append(Vn);araw.append(Au)
    fenv.append(np.linalg.norm(np.diff(Vn,axis=0),axis=1));venv.append(np.abs(Au).mean(1))
    y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
pca=PCA(20).fit(np.vstack([v[::3] for v in vraw]))
vis=[];aud=[];cov=[];coord=[]
for i in range(len(y)):
    Z=pca.transform(vraw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);vis.append(resamp(Zz))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);aud.append(resamp(Aa))
    c,_=ledoit_wolf(Z);cov.append(c+1e-3*np.eye(20));coord.append(coord_feats(fenv[i],venv[i]))
out.append(run(np.array(vis),np.array(aud),cov,coord,y,20,araw[0].shape[1],'D-Vlog'))
print('\n판정: AV>visual & full 최선이면 → coordination-aware 경량 AV = 헤드룸 실증 + 우리 프레임.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp79_av.csv','w') as f_:
    f_.write('corpus,n,visual,AV,full\n')
    for r in out:f_.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp79_av.csv',flush=True)
