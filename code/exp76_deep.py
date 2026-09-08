"""
exp76 — 딥러닝 축 개시(헤드룸): 1D-CNN 시계열 + 우리 coupling 하이브리드.
큰 코퍼스(LMVD AU / D-Vlog landmark). 비교: coupling-LR / CNN / CNN+coupling(하이브리드).
질문: 딥모델이 coupling 천장(LMVD .74/D-Vlog .63) 넘나 + 하이브리드가 최선인가(=강화 여지).
결과: results/exp76_deep.csv
"""
import numpy as np, warnings, csv, glob
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); dev='cuda'; T=256; EP=45
torch.manual_seed(0); np.random.seed(0)
def resamp_seq(X,t=T):
    n=len(X)
    if n==t:return X
    idx=np.linspace(0,n-1,t)
    return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def cov_of(X):
    seg=(X-X.mean(0))/(X.std(0)+1e-6);c,_=ledoit_wolf(seg);return c+1e-4*np.eye(X.shape[1])
class Net(nn.Module):
    def __init__(self,C,aux=0):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv1d(C,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(128,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Linear(128+aux,64),nn.ReLU(),nn.Dropout(0.4),nn.Linear(64,1))
    def forward(self,x,a=None):
        h=self.cnn(x).squeeze(-1)
        if a is not None: h=torch.cat([h,a],1)
        return self.head(h).squeeze(-1)
def train_eval(Xtr,atr,ytr,Xte,ate,yte,C,aux):
    net=Net(C,aux).to(dev)
    opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    pw=torch.tensor([(ytr==0).sum()/max(1,(ytr==1).sum())],dtype=torch.float32,device=dev)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xtr_t=torch.tensor(Xtr,dtype=torch.float32,device=dev).transpose(1,2)
    Xte_t=torch.tensor(Xte,dtype=torch.float32,device=dev).transpose(1,2)
    atr_t=torch.tensor(atr,dtype=torch.float32,device=dev) if aux else None
    ate_t=torch.tensor(ate,dtype=torch.float32,device=dev) if aux else None
    ytr_t=torch.tensor(ytr,dtype=torch.float32,device=dev)
    best=0.5;n=len(Xtr);bs=64
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs]
            opt.zero_grad()
            out=net(Xtr_t[b], atr_t[b] if aux else None)
            loss=lossf(out,ytr_t[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            p=torch.sigmoid(net(Xte_t, ate_t if aux else None)).cpu().numpy()
        try:au=roc_auc_score(yte,p);best=max(best,au)
        except:pass
    return best
def run(seqs, covs, y, C, name):
    y=np.array(y);covs=np.array(covs)
    print(f'\n===== {name} (n={len(y)}, 우울{int(y.sum())}, C={C}) =====',flush=True)
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    res={'coupling':[],'CNN':[],'hybrid':[]}
    for fold,(tr,te) in enumerate(skf.split(seqs,y)):
        Xtr=seqs[tr];Xte=seqs[te]
        # coupling-LR
        Ts=TangentSpace(metric='riemann').fit(covs[tr]);Ztr=Ts.transform(covs[tr]);Zte=Ts.transform(covs[te])
        sc=StandardScaler().fit(Ztr);lr=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Ztr),y[tr])
        res['coupling'].append(roc_auc_score(y[te],lr.decision_function(sc.transform(Zte))))
        # CNN
        res['CNN'].append(train_eval(Xtr,None,y[tr],Xte,None,y[te],C,0))
        # hybrid: coupling tangent as aux (train-fit)
        ascaler=StandardScaler().fit(Ztr)
        res['hybrid'].append(train_eval(Xtr,ascaler.transform(Ztr),y[tr],Xte,ascaler.transform(Zte),y[te],C,Ztr.shape[1]))
        print(f'  fold{fold}: coupling={res["coupling"][-1]:.3f} CNN={res["CNN"][-1]:.3f} hybrid={res["hybrid"][-1]:.3f}',flush=True)
    for k in res: print(f'  {k:10s} 평균 AUC={np.mean(res[k]):.3f}±{np.std(res[k]):.3f}',flush=True)
    return [name,len(y),np.mean(res['coupling']),np.mean(res['CNN']),np.mean(res['hybrid'])]
out=[]
seqs=[];covs=[];y=[]
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423):return 1
    if (602<=i<=1116) or (1425<=i<=1824):return 0
    return None
print('LMVD 로딩...',flush=True)
for f in sorted(glob.glob(str(V/'*.csv'))):
    l=lab(int(Path(f).stem))
    if l is None:continue
    h=[x.strip() for x in open(f).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AU]
    except:continue
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    if len(fe)<60:continue
    X=np.nan_to_num(np.array(fe));Xz=(X-X.mean(0))/(X.std(0)+1e-6)
    seqs.append(resamp_seq(Xz));covs.append(cov_of(X));y.append(l)
out.append(run(np.array(seqs),covs,y,len(AU),'LMVD'))
# D-Vlog
seqs=[];covs=[];y=[]
from sklearn.decomposition import PCA
R=B/'D-Vlog'
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
raw=[]
print('D-Vlog 로딩...',flush=True)
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:Vv=np.load(fv)
    except:continue
    if Vv.ndim!=2 or Vv.shape[0]<60 or Vv.shape[1]!=136:continue
    raw.append(nf(Vv));y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
pca=PCA(20).fit(np.vstack([v[::3] for v in raw]))
for Vn in raw:
    Z=pca.transform(Vn);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6)
    seqs.append(resamp_seq(Zz));c,_=ledoit_wolf(Z);covs.append(c+1e-3*np.eye(20))
out.append(run(np.array(seqs),covs,y,20,'D-Vlog'))
print('\n판정: CNN or hybrid이 coupling 넘고 hybrid이 최선이면 → 딥 축 헤드룸 확보(강화 여지 있음).',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp76_deep.csv','w') as f_:
    f_.write('corpus,n,coupling,CNN,hybrid\n')
    for r in out:f_.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp76_deep.csv',flush=True)
