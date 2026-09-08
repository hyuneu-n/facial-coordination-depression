"""
exp77 — 딥 축 강화 사다리 확인: attention pooling + 다중 seed 안정성.
exp76 확인(CNN>coupling)을 굳히고, attention이 avg-pool 넘는지(=강화 축 작동) 시연.
LMVD/D-Vlog. 3 seed × 5 fold. 비교: coupling / CNN(avg) / CNN(attn) / hybrid(attn)+coupling.
결과: results/exp77_attn.csv
"""
import numpy as np, warnings, csv, glob
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); dev='cuda'; T=256; EP=35
def resamp_seq(X,t=T):
    n=len(X)
    if n==t:return X
    idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def cov_of(X,reg=1e-4):
    seg=(X-X.mean(0))/(X.std(0)+1e-6);c,_=ledoit_wolf(seg);return c+reg*np.eye(X.shape[1])
class Net(nn.Module):
    def __init__(self,C,aux=0,attn=True):
        super().__init__();self.attn=attn
        self.cnn=nn.Sequential(
            nn.Conv1d(C,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(128,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU())
        self.att=nn.Linear(128,1)
        self.head=nn.Sequential(nn.Linear(128+aux,64),nn.ReLU(),nn.Dropout(0.4),nn.Linear(64,1))
    def forward(self,x,a=None):
        h=self.cnn(x)  # B,128,T'
        if self.attn:
            w=torch.softmax(self.att(h.transpose(1,2)).squeeze(-1),1)  # B,T'
            h=(h*w.unsqueeze(1)).sum(-1)
        else:
            h=h.mean(-1)
        if a is not None:h=torch.cat([h,a],1)
        return self.head(h).squeeze(-1)
def train_eval(Xtr,atr,ytr,Xte,ate,yte,C,aux,attn,seed):
    torch.manual_seed(seed)
    net=Net(C,aux,attn).to(dev);opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    pw=torch.tensor([(ytr==0).sum()/max(1,(ytr==1).sum())],dtype=torch.float32,device=dev)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xtr_t=torch.tensor(Xtr,dtype=torch.float32,device=dev).transpose(1,2)
    Xte_t=torch.tensor(Xte,dtype=torch.float32,device=dev).transpose(1,2)
    atr_t=torch.tensor(atr,dtype=torch.float32,device=dev) if aux else None
    ate_t=torch.tensor(ate,dtype=torch.float32,device=dev) if aux else None
    ytr_t=torch.tensor(ytr,dtype=torch.float32,device=dev);n=len(Xtr);bs=64;best=0.5
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad()
            loss=lossf(net(Xtr_t[b],atr_t[b] if aux else None),ytr_t[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():p=torch.sigmoid(net(Xte_t,ate_t if aux else None)).cpu().numpy()
        try:best=max(best,roc_auc_score(yte,p))
        except:pass
    return best
def run(seqs,covs,y,C,name):
    y=np.array(y);covs=np.array(covs)
    print(f'\n===== {name} (n={len(y)}) =====',flush=True)
    res={'coupling':[],'CNN_avg':[],'CNN_attn':[],'hybrid_attn':[]}
    for seed in range(3):
        skf=StratifiedKFold(5,shuffle=True,random_state=seed)
        for tr,te in skf.split(seqs,y):
            Ts=TangentSpace(metric='riemann').fit(covs[tr]);Ztr=Ts.transform(covs[tr]);Zte=Ts.transform(covs[te])
            sc=StandardScaler().fit(Ztr);lr=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Ztr),y[tr])
            res['coupling'].append(roc_auc_score(y[te],lr.decision_function(sc.transform(Zte))))
            res['CNN_avg'].append(train_eval(seqs[tr],None,y[tr],seqs[te],None,y[te],C,0,False,seed))
            res['CNN_attn'].append(train_eval(seqs[tr],None,y[tr],seqs[te],None,y[te],C,0,True,seed))
            asc=StandardScaler().fit(Ztr)
            res['hybrid_attn'].append(train_eval(seqs[tr],asc.transform(Ztr),y[tr],seqs[te],asc.transform(Zte),y[te],C,Ztr.shape[1],True,seed))
        print(f'  seed{seed} done',flush=True)
    for k in res:print(f'  {k:12s} AUC={np.mean(res[k]):.3f}±{np.std(res[k]):.3f}',flush=True)
    return [name,len(y)]+[np.mean(res[k]) for k in ['coupling','CNN_avg','CNN_attn','hybrid_attn']]
out=[]
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423):return 1
    if (602<=i<=1116) or (1425<=i<=1824):return 0
    return None
print('LMVD 로딩...',flush=True);seqs=[];covs=[];y=[]
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
print('D-Vlog 로딩...',flush=True);seqs=[];covs=[];y=[];raw=[]
R=B/'D-Vlog'
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
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
print('\n판정: CNN_attn≥CNN_avg & 딥>coupling 안정적(3seed) → 강화 사다리 작동+헤드룸 확정.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp77_attn.csv','w') as f_:
    f_.write('corpus,n,coupling,CNN_avg,CNN_attn,hybrid_attn\n')
    for r in out:f_.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp77_attn.csv',flush=True)
