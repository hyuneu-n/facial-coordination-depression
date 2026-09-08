"""
exp81 — 비전-네이티브: 얼굴 동역학을 이미지로 인코딩 → 2D-CNN 패턴인식 (현은 비전 강점).
인코딩: (A) recurrence image(프레임 자기거리 행렬 T×T=동역학 지문) (B) channel×time 히트맵.
D-Vlog 공식 split서 2D-CNN F1/AUC → exp80(1D CNN, F1 0.727)과 비교.
결과: results/exp81_image.csv
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog'); dev='cuda'; T=128; SZ=96; EP=60
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
def resamp(X,t=T):
    n=len(X);idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def recurrence_img(seq):  # (T,C)->(SZ,SZ) 자기거리 행렬 리사이즈
    d=np.linalg.norm(seq[:,None,:]-seq[None,:,:],axis=2)  # T,T
    d=d/(d.max()+1e-6)
    t=torch.tensor(d,dtype=torch.float32)[None,None]
    return Fn.interpolate(t,size=(SZ,SZ),mode='bilinear',align_corners=False)[0,0].numpy()
def cht_img(seq):  # channel×time -> (C, SZ) 리사이즈, C=20
    t=torch.tensor(seq.T,dtype=torch.float32)[None,None]  # 1,1,C,T
    return Fn.interpolate(t,size=(SZ,SZ),mode='bilinear',align_corners=False)[0,0].numpy()
class CNN2D(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(
            nn.Conv2d(2,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32,1))
    def forward(self,x):return self.head(self.net(x).flatten(1)).squeeze(-1)
rows=list(csv.DictReader(open(R/'labels.csv')))
imgs=[];Y=[];F=[]
raw=[]
for r in rows:
    idx=r['index'].strip();fold=r.get('fold','').strip().lower()
    if 'val' in fold:fold='valid'
    if fold not in('train','valid','test'):continue
    fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:V=np.load(fv)
    except:continue
    if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136:continue
    raw.append(nf(V));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0);F.append(fold)
Y=np.array(Y);F=np.array(F)
pca=PCA(20).fit(np.vstack([raw[i][::3] for i in range(len(raw)) if F[i]=='train']))
for V in raw:
    Z=pca.transform(V);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);s=resamp(Zz)
    imgs.append(np.stack([recurrence_img(s),cht_img(s)],0))  # 2ch image
imgs=np.array(imgs,dtype=np.float32)
print('img shape',imgs.shape,'fold',{k:int((F==k).sum()) for k in ['train','valid','test']},flush=True)
tr=F=='train';va=F=='valid';te=F=='test'
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
res=[]
for seed in range(5):
    torch.manual_seed(seed);net=CNN2D().to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xtr=T3(imgs[tr]);ytr=T3(Y[tr]);Xva=T3(imgs[va]);Xte=T3(imgs[te]);n=tr.sum();bs=64;bestva=0;bestp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Xtr[b]),ytr[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            pv=torch.sigmoid(net(Xva)).cpu().numpy();pt=torch.sigmoid(net(Xte)).cpu().numpy()
        try:
            av=roc_auc_score(Y[va],pv)
            if av>bestva:bestva=av;bestp=pt
        except:pass
    if bestp is not None:res.append((roc_auc_score(Y[te],bestp),f1_score(Y[te],(bestp>0.5).astype(int))))
res=np.array(res)
print(f'\n===== D-Vlog 공식 test — 비전(2D-CNN on 동역학 이미지) =====',flush=True)
print(f'  AUC={res[:,0].mean():.3f}±{res[:,0].std():.3f}  F1={res[:,1].mean():.3f}±{res[:,1].std():.3f}',flush=True)
print(f'  [비교] exp80 1D-CNN AV+coupling F1 0.727 | D-Vlog 원논문 0.63',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp81_image.csv','w') as f:
    f.write('metric,mean,std\n');f.write(f'AUC,{res[:,0].mean():.4f},{res[:,0].std():.4f}\nF1,{res[:,1].mean():.4f},{res[:,1].std():.4f}\n')
print('DONE → exp81_image.csv',flush=True)
