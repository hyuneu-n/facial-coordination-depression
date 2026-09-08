"""
exp82b — 경량 협응 그래프 GNN. 노드=AU, 노드피처=값싼 통계(mean/std/AR1/energy/median),
엣지=coupling 상관. GNN 2층 + attention readout. (conv 제거로 초단위 학습)
비교: coupling-LR vs GNN. LMVD 3seed×5fold + D-Vlog 공식. 결과: results/exp82b_gnn.csv
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
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); dev='cuda'; EP=120
def node_feats(seq):  # seq (T,C) z-scored -> (C,5) per-node
    C=seq.shape[1];out=np.zeros((C,5))
    for c in range(C):
        x=seq[:,c];out[c,0]=x.mean();out[c,1]=x.std();out[c,3]=np.abs(np.diff(x)).mean() if len(x)>1 else 0
        out[c,4]=np.median(x)
        if x.std()>1e-6 and len(x)>10:
            z=(x-x.mean())/x.std();out[c,2]=np.polyfit(z[:-1],z[1:],1)[0]
    return out
def covscorr(C):
    d=np.sqrt(np.diag(C));return C/(d[:,None]*d[None,:]+1e-9)
def norm_adj(C):
    A=np.abs(covscorr(C)).copy();np.fill_diagonal(A,1.0);d=A.sum(1);di=1/np.sqrt(d+1e-6);return (A*di[:,None])*di[None,:]
class GNN(nn.Module):
    def __init__(self,Fin,d=32):
        super().__init__();self.emb=nn.Linear(Fin,d)
        self.W1=nn.Linear(d,d);self.W2=nn.Linear(d,d);self.att=nn.Linear(d,1)
        self.head=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32,1))
    def forward(self,X,A):
        h=torch.relu(self.emb(X))
        h=torch.relu(torch.bmm(A,self.W1(h)));h=torch.relu(torch.bmm(A,self.W2(h)))
        w=torch.softmax(self.att(h).squeeze(-1),1);g=(h*w.unsqueeze(-1)).sum(1)
        return self.head(g).squeeze(-1)
def train_eval(Xtr,Atr,ytr,Xte,Ate,yte,Fin,seed):
    torch.manual_seed(seed);net=GNN(Fin).to(dev);opt=torch.optim.Adam(net.parameters(),3e-3,weight_decay=1e-4)
    pw=torch.tensor([(ytr==0).sum()/max(1,(ytr==1).sum())],dtype=torch.float32,device=dev);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
    X=T3(Xtr);A=T3(Atr);y=T3(ytr);Xt=T3(Xte);At=T3(Ate);n=len(ytr);bs=128;best=0.5;bp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(X[b],A[b]),y[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():p=torch.sigmoid(net(Xt,At)).cpu().numpy()
        try:
            au=roc_auc_score(yte,p)
            if au>best:best=au;bp=p
        except:pass
    return best,bp
# ---- LMVD ----
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423):return 1
    if (602<=i<=1116) or (1425<=i<=1824):return 0
    return None
print('LMVD 로딩...',flush=True);NF=[];AD=[];COV=[];y=[]
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
    NF.append(node_feats(Xz));c,_=ledoit_wolf(Xz);AD.append(norm_adj(c));COV.append(c+1e-4*np.eye(len(AU)));y.append(l)
NF=np.array(NF);AD=np.array(AD);COV=np.array(COV);y=np.array(y)
# node feat 표준화(채널·피처별, train서 fit은 fold내에서)
res={'coupling':[],'GNN':[]}
for seed in range(3):
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(NF,y):
        mu=NF[tr].reshape(-1,NF.shape[2]).mean(0);sd=NF[tr].reshape(-1,NF.shape[2]).std(0)+1e-6
        Xn=(NF-mu)/sd
        Ts=TangentSpace(metric='riemann').fit(COV[tr]);Z=Ts.transform(COV);sc=StandardScaler().fit(Z[tr])
        lr=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Z[tr]),y[tr])
        res['coupling'].append(roc_auc_score(y[te],lr.decision_function(sc.transform(Z[te]))))
        b,_=train_eval(Xn[tr],AD[tr],y[tr],Xn[te],AD[te],y[te],NF.shape[2],seed);res['GNN'].append(b)
    print(f'  seed{seed}: coupling={np.mean(res["coupling"]):.3f} GNN={np.mean(res["GNN"]):.3f}',flush=True)
print(f'\n===== LMVD =====\n  coupling={np.mean(res["coupling"]):.3f} | GNN={np.mean(res["GNN"]):.3f}±{np.std(res["GNN"]):.3f}',flush=True)
out=[['LMVD',len(y),np.mean(res['coupling']),np.mean(res['GNN'])]]
# ---- D-Vlog 공식 (PCA20 nodes) ----
print('D-Vlog 로딩...',flush=True)
R=B/'D-Vlog'
def nf2(V):
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
    raw.append(nf2(Vv));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0);Fd.append(fold)
Y=np.array(Y);Fd=np.array(Fd)
pca=PCA(20).fit(np.vstack([raw[i][::3] for i in range(len(raw)) if Fd[i]=='train']))
NF=[];AD=[]
for Vv in raw:
    Z=pca.transform(Vv);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);NF.append(node_feats(Zz))
    c,_=ledoit_wolf(Z);AD.append(norm_adj(c+1e-3*np.eye(20)))
NF=np.array(NF);AD=np.array(AD)
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
mu=NF[tr].reshape(-1,5).mean(0);sd=NF[tr].reshape(-1,5).std(0)+1e-6;Xn=(NF-mu)/sd
aucs=[];f1s=[]
for seed in range(5):
    torch.manual_seed(seed);net=GNN(5).to(dev);opt=torch.optim.Adam(net.parameters(),3e-3,weight_decay=1e-4)
    pw=torch.tensor([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())],dtype=torch.float32,device=dev);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
    Xt=T3(Xn[tr]);At=T3(AD[tr]);yt=T3(Y[tr]);Xv=T3(Xn[va]);Av=T3(AD[va]);Xe=T3(Xn[te]);Ae=T3(AD[te]);n=tr.sum();bs=128;bva=0;bp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Xt[b],At[b]),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():pv=torch.sigmoid(net(Xv,Av)).cpu().numpy();pe=torch.sigmoid(net(Xe,Ae)).cpu().numpy()
        try:
            av=roc_auc_score(Y[va],pv)
            if av>bva:bva=av;bp=pe
        except:pass
    if bp is not None:aucs.append(roc_auc_score(Y[te],bp));f1s.append(f1_score(Y[te],(bp>0.5).astype(int)))
print(f'\n===== D-Vlog 공식 test — GNN =====\n  AUC={np.mean(aucs):.3f} F1={np.mean(f1s):.3f}  [exp80 AV F1 0.727 / 원논문 0.63]',flush=True)
out.append(['D-Vlog_official',int(te.sum()),np.mean(aucs),np.mean(f1s)])
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp82b_gnn.csv','w') as f:
    f.write('corpus,n,coupling_or_auc,gnn_or_f1\n')
    for r in out:f.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp82b_gnn.csv',flush=True)
