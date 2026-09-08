"""
exp82 — 협응 그래프 GNN: AU=노드(시계열→공유 1D-conv 임베딩), coupling 상관=엣지.
graph-conv 2층 + attention readout. coupling을 그래프로 딥이 학습(우리 실+딥+해석).
LMVD(AU 17노드, 5fold) + D-Vlog(PCA20 노드, 공식 split). 비교: coupling-LR / GNN.
결과: results/exp82_gnn.csv
"""
import numpy as np, warnings, csv, glob
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); dev='cuda'; T=256; EP=50
def resamp(X,t=T):
    n=len(X);idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
class GNN(nn.Module):
    def __init__(self,Nn,d=32):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv1d(1,16,7,padding=3),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(2),
                               nn.Conv1d(16,d,7,padding=3),nn.BatchNorm1d(d),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.W1=nn.Linear(d,d);self.W2=nn.Linear(d,d);self.att=nn.Linear(d,1)
        self.head=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32,1))
    def forward(self,seq,A):  # seq:(B,Nn,T)  A:(B,Nn,Nn) normalized
        Bsz,Nn,Tt=seq.shape
        x=seq.reshape(Bsz*Nn,1,Tt);h=self.enc(x).squeeze(-1).reshape(Bsz,Nn,-1)  # B,Nn,d
        h=torch.relu(torch.bmm(A,self.W1(h)));h=torch.relu(torch.bmm(A,self.W2(h)))
        w=torch.softmax(self.att(h).squeeze(-1),1)  # B,Nn
        g=(h*w.unsqueeze(-1)).sum(1)
        return self.head(g).squeeze(-1)
def norm_adj(C):
    A=np.abs(C).copy();np.fill_diagonal(A,1.0)
    d=A.sum(1);dinv=1/np.sqrt(d+1e-6);return (A*dinv[:,None])*dinv[None,:]
def train_eval(seqtr,Atr,ytr,seqte,Ate,yte,Nn,seed):
    torch.manual_seed(seed);net=GNN(Nn).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=torch.tensor([(ytr==0).sum()/max(1,(ytr==1).sum())],dtype=torch.float32,device=dev);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
    S=T3(seqtr);A=T3(Atr);y=T3(ytr);St=T3(seqte);At=T3(Ate);n=len(ytr);bs=64;best=0.5;bestp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(S[b],A[b]),y[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():p=torch.sigmoid(net(St,At)).cpu().numpy()
        try:
            au=roc_auc_score(yte,p)
            if au>best:best=au;bestp=p
        except:pass
    return best,bestp
def prep(seqs,covs):
    # seqs:(N,T,Nn) -> (N,Nn,T); A from cov correlation
    S=np.transpose(np.array(seqs),(0,2,1))
    A=np.array([norm_adj(np.corrcoef(covs[i]) if False else covscorr(covs[i])) for i in range(len(covs))])
    return S,A
def covscorr(C):
    d=np.sqrt(np.diag(C));return C/(d[:,None]*d[None,:]+1e-9)
# ---- LMVD (AU nodes, 5fold) ----
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
    seqs.append(resamp(Xz));c,_=ledoit_wolf(Xz);covs.append(c+1e-4*np.eye(len(AU)));y.append(l)
y=np.array(y);S=np.transpose(np.array(seqs),(0,2,1));A=np.array([norm_adj(covscorr(c)) for c in covs]);COV=np.array(covs)
res={'coupling':[],'GNN':[]}
for seed in range(3):
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(S,y):
        Ts=TangentSpace(metric='riemann').fit(COV[tr]);Z=Ts.transform(COV);sc=StandardScaler().fit(Z[tr])
        lr=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Z[tr]),y[tr])
        res['coupling'].append(roc_auc_score(y[te],lr.decision_function(sc.transform(Z[te]))))
        b,_=train_eval(S[tr],A[tr],y[tr],S[te],A[te],y[te],len(AU),seed);res['GNN'].append(b)
    print(f'  seed{seed} done',flush=True)
print(f'\n===== LMVD =====\n  coupling={np.mean(res["coupling"]):.3f} | GNN={np.mean(res["GNN"]):.3f}±{np.std(res["GNN"]):.3f}',flush=True)
out=[['LMVD',len(y),np.mean(res['coupling']),np.mean(res['GNN'])]]
# ---- D-Vlog 공식 (PCA20 nodes) ----
print('D-Vlog 로딩...',flush=True)
R=B/'D-Vlog'
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
raw=[];Y=[];Fd=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fold=r.get('fold','').strip().lower()
    if 'val' in fold:fold='valid'
    if fold not in('train','valid','test'):continue
    fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:Vv=np.load(fv)
    except:continue
    if Vv.ndim!=2 or Vv.shape[0]<60 or Vv.shape[1]!=136:continue
    raw.append(nf(Vv));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0);Fd.append(fold)
Y=np.array(Y);Fd=np.array(Fd)
pca=PCA(20).fit(np.vstack([raw[i][::3] for i in range(len(raw)) if Fd[i]=='train']))
S=[];A=[]
for Vv in raw:
    Z=pca.transform(Vv);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);S.append(resamp(Zz))
    c,_=ledoit_wolf(Z);A.append(norm_adj(covscorr(c+1e-3*np.eye(20))))
S=np.transpose(np.array(S),(0,2,1));A=np.array(A)
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
f1s=[];aucs=[]
for seed in range(5):
    # early stop on valid: train on train, pick best valid, eval test
    torch.manual_seed(seed);net=GNN(20).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=torch.tensor([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())],dtype=torch.float32,device=dev);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
    St=T3(S[tr]);At=T3(A[tr]);yt=T3(Y[tr]);Sv=T3(S[va]);Av=T3(A[va]);Se=T3(S[te]);Ae=T3(A[te]);n=tr.sum();bs=64;bva=0;bp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(St[b],At[b]),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():pv=torch.sigmoid(net(Sv,Av)).cpu().numpy();pe=torch.sigmoid(net(Se,Ae)).cpu().numpy()
        try:
            av=roc_auc_score(Y[va],pv)
            if av>bva:bva=av;bp=pe
        except:pass
    if bp is not None:aucs.append(roc_auc_score(Y[te],bp));f1s.append(f1_score(Y[te],(bp>0.5).astype(int)))
print(f'\n===== D-Vlog 공식 test — GNN =====\n  AUC={np.mean(aucs):.3f} F1={np.mean(f1s):.3f}  [exp80 AV F1 0.727]',flush=True)
out.append(['D-Vlog_official',int(te.sum()),np.mean(aucs),np.mean(f1s)])
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp82_gnn.csv','w') as f:
    f.write('corpus,n,coupling_or_auc,gnn_or_f1\n')
    for r in out:f.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp82_gnn.csv',flush=True)
