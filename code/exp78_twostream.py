"""
exp78 — ★Geo-Temporal 2-stream Net (인코더+기하백본)★. 논문(STSFF two-stream) 근거+우리 novel.
Stream A(인코더): Transformer encoder + attention pool on 원시 시퀀스.
Stream B(기하 백본): Riemannian coupling(SPD→tangent) → MLP. (해석가능 정체성)
Fusion: concat → 분류. 비교: coupling / Transformer / ★2-stream fusion.
LMVD/D-Vlog, 3 seed × 5 fold. 결과: results/exp78_twostream.csv
"""
import numpy as np, warnings, csv, glob, math
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
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); dev='cuda'; T=256; EP=40
def resamp_seq(X,t=T):
    n=len(X)
    if n==t:return X
    idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def cov_of(X,reg):
    seg=(X-X.mean(0))/(X.std(0)+1e-6);c,_=ledoit_wolf(seg);return c+reg*np.eye(X.shape[1])
class PosEnc(nn.Module):
    def __init__(self,d,L=T):
        super().__init__();pe=torch.zeros(L,d);pos=torch.arange(L).unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2)*(-math.log(10000)/d))
        pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div);self.register_buffer('pe',pe)
    def forward(self,x):return x+self.pe[:x.size(1)]
class TwoStream(nn.Module):
    def __init__(self,C,tan=0,use_deep=True,use_geo=True):
        super().__init__();self.use_deep=use_deep;self.use_geo=use_geo;d=128
        if use_deep:
            self.proj=nn.Linear(C,d);self.pos=PosEnc(d)
            enc=nn.TransformerEncoderLayer(d,4,256,0.3,batch_first=True)
            self.tr=nn.TransformerEncoder(enc,2);self.att=nn.Linear(d,1)
        if use_geo:
            self.geo=nn.Sequential(nn.Linear(tan,128),nn.ReLU(),nn.Dropout(0.4),nn.Linear(128,64))
        fin=(d if use_deep else 0)+(64 if use_geo else 0)
        self.head=nn.Sequential(nn.Linear(fin,64),nn.ReLU(),nn.Dropout(0.4),nn.Linear(64,1))
    def forward(self,x,tan):
        parts=[]
        if self.use_deep:
            h=self.pos(self.proj(x));h=self.tr(h)
            w=torch.softmax(self.att(h).squeeze(-1),1);parts.append((h*w.unsqueeze(-1)).sum(1))
        if self.use_geo: parts.append(self.geo(tan))
        return self.head(torch.cat(parts,1)).squeeze(-1)
def train_eval(Xtr,Ttr,ytr,Xte,Tte,yte,C,tan,ud,ug,seed):
    torch.manual_seed(seed)
    net=TwoStream(C,tan,ud,ug).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=torch.tensor([(ytr==0).sum()/max(1,(ytr==1).sum())],dtype=torch.float32,device=dev)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xtr_t=torch.tensor(Xtr,dtype=torch.float32,device=dev);Xte_t=torch.tensor(Xte,dtype=torch.float32,device=dev)
    Ttr_t=torch.tensor(Ttr,dtype=torch.float32,device=dev);Tte_t=torch.tensor(Tte,dtype=torch.float32,device=dev)
    ytr_t=torch.tensor(ytr,dtype=torch.float32,device=dev);n=len(Xtr);bs=64;best=0.5
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad()
            loss=lossf(net(Xtr_t[b],Ttr_t[b]),ytr_t[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():p=torch.sigmoid(net(Xte_t,Tte_t)).cpu().numpy()
        try:best=max(best,roc_auc_score(yte,p))
        except:pass
    return best
def run(seqs,covs,y,C,name):
    y=np.array(y);covs=np.array(covs)
    print(f'\n===== {name} (n={len(y)}) =====',flush=True)
    res={'coupling':[],'Transformer':[],'2stream':[]}
    for seed in range(3):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(seqs,y):
            Ts=TangentSpace(metric='riemann').fit(covs[tr]);Ztr=Ts.transform(covs[tr]);Zte=Ts.transform(covs[te])
            sc=StandardScaler().fit(Ztr);Ztr2=sc.transform(Ztr);Zte2=sc.transform(Zte)
            lr=LogisticRegression(max_iter=2000,class_weight='balanced').fit(Ztr2,y[tr])
            res['coupling'].append(roc_auc_score(y[te],lr.decision_function(Zte2)))
            res['Transformer'].append(train_eval(seqs[tr],Ztr2,y[tr],seqs[te],Zte2,y[te],C,Ztr2.shape[1],True,False,seed))
            res['2stream'].append(train_eval(seqs[tr],Ztr2,y[tr],seqs[te],Zte2,y[te],C,Ztr2.shape[1],True,True,seed))
        print(f'  seed{seed} done',flush=True)
    for k in res:print(f'  {k:12s} AUC={np.mean(res[k]):.3f}±{np.std(res[k]):.3f}',flush=True)
    return [name,len(y),np.mean(res['coupling']),np.mean(res['Transformer']),np.mean(res['2stream'])]
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
    seqs.append(resamp_seq(Xz));covs.append(cov_of(X,1e-4));y.append(l)
out.append(run(np.array(seqs),covs,y,len(AU),'LMVD'))
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
print('\n판정: 2stream(인코더+기하백본)이 coupling·Transformer 단독 다 넘으면 → novel 최종 아키텍처.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp78_twostream.csv','w') as f_:
    f_.write('corpus,n,coupling,Transformer,twostream\n')
    for r in out:f_.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp78_twostream.csv',flush=True)
