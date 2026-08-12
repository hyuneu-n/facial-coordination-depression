"""
exp35 — C3: DAIC-WOZ 모델비교. tangent-LR(baseline) / SPDNet / VNN / sparse-L1.
CMDC와 동일 결론(소규모 경량기하 우세)이 DAIC서도 재현되나. n=94, 14 AU 공분산.
Transformer는 소규모 과적합(CMDC서 확인)이라 제외 — 발표에 그 이유로 명시.
결과: results/model_daic.csv
"""
import numpy as np, warnings, csv
from pathlib import Path
import torch, torch.nn as nn, geoopt
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
NA=len(AU); SEEDS=list(range(10)); SEEDS5=[42,1,2,3,4]
dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
NEG=['feel_lately','depression_diagnosed','feelguilty','regret','feelbadly','last_argument','control_temper']
def dseries(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None,None
    h=[x.strip() for x in open(p).readline().split(',')];ti,oi=h.index('timestamp'),h.index('success');ai=[h.index(c) for c in AU]
    ts,fe=[],[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            ts.append(float(v[ti]));fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(ts),np.array(fe)
def dtrans(pid):
    p=D/f'{pid}_TRANSCRIPT.csv';rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try:rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except:pass
    return rows
covs=[];X0=[];y=[]
for pid,l in dl.items():
    ts,au=dseries(pid)
    if ts is None:continue
    segs=[]
    for st,sp,spk,val in dtrans(pid):
        if spk=='Ellie' and any(val.startswith(t) for t in NEG):
            m=(ts>=sp)&(ts<sp+8.0)
            if m.sum()>=8:segs.append(au[m])
    if not segs:continue
    seg=np.vstack(segs);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6)
    if len(seg)<15:continue
    c,_=ledoit_wolf(seg);covs.append(c);X0.append(seg.mean(0));y.append(l)
covs=np.array(covs);X0=np.array(X0,dtype=np.float32);y=np.array(y)
print(f'DAIC anchor: n={len(y)} MDD{int(y.sum())}/HC{int((y==0).sum())} cov{covs.shape}',flush=True)
res={}

# --- tangent-LR baseline ---
a=[]
for sd in SEEDS:
    skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
    for tr,te in skf.split(covs,y):
        clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
        clf.fit(covs[tr],y[tr]);pb[te]=clf.decision_function(covs[te])
    a.append(roc_auc_score(y,pb))
res['tangent_LR']=(np.mean(a),np.std(a)); print(f'  tangent-LR AUC={np.mean(a):.3f}±{np.std(a):.3f}',flush=True)

# --- sparse L1 ---
ts=TangentSpace(metric='riemann');T=ts.fit_transform(covs); a=[]
for sd in SEEDS:
    skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
    for tr,te in skf.split(T,y):
        scaler=StandardScaler().fit(T[tr])
        clf=LogisticRegression(max_iter=5000,class_weight='balanced',penalty='l1',solver='liblinear',C=0.5)
        clf.fit(scaler.transform(T[tr]),y[tr]);pb[te]=clf.decision_function(scaler.transform(T[te]))
    a.append(roc_auc_score(y,pb))
res['sparse_L1']=(np.mean(a),np.std(a)); print(f'  sparse-L1  AUC={np.mean(a):.3f}±{np.std(a):.3f}',flush=True)

# --- SPDNet ---
torch.set_default_dtype(torch.float64)
class BiMap(nn.Module):
    def __init__(s,n_in,n_out):
        super().__init__();w=torch.empty(n_in,n_out);nn.init.orthogonal_(w)
        s.W=geoopt.ManifoldParameter(w,manifold=geoopt.Stiefel())
    def forward(s,X):return torch.einsum('ji,bjk,kl->bil',s.W,X,s.W)
class ReEig(nn.Module):
    def __init__(s,eps=1e-4):super().__init__();s.eps=eps
    def forward(s,X):
        X=(X+X.transpose(-1,-2))/2;val,vec=torch.linalg.eigh(X);val=torch.clamp(val,min=s.eps)
        return vec@torch.diag_embed(val)@vec.transpose(-1,-2)
class LogEig(nn.Module):
    def forward(s,X):
        X=(X+X.transpose(-1,-2))/2;val,vec=torch.linalg.eigh(X);val=torch.clamp(val,min=1e-6)
        return vec@torch.diag_embed(torch.log(val))@vec.transpose(-1,-2)
class SPDNet(nn.Module):
    def __init__(s,n=NA,h=8):
        super().__init__();s.bimap=BiMap(n,h);s.reeig=ReEig();s.logeig=LogEig()
        s.bn=nn.BatchNorm1d(h*h);s.fc=nn.Sequential(nn.Dropout(0.4),nn.Linear(h*h,1))
    def forward(s,X):
        X=s.logeig(s.reeig(s.bimap(X)));v=s.bn(X.reshape(X.shape[0],-1));return s.fc(v).squeeze(-1)
def run_spd(seed):
    torch.manual_seed(seed);np.random.seed(seed)
    skf=StratifiedKFold(5,shuffle=True,random_state=seed);proba=np.zeros(len(y))
    for tr,te in skf.split(covs,y):
        m=SPDNet().to(DEV);opt=geoopt.optim.RiemannianAdam(m.parameters(),lr=5e-3,weight_decay=1e-3)
        pw=torch.tensor([(y[tr]==0).sum()/max((y[tr]==1).sum(),1)],dtype=torch.float64).to(DEV)
        lf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt=torch.tensor(covs[tr]).to(DEV);yt=torch.tensor(y[tr],dtype=torch.float64).to(DEV)
        best=None;bl=1e9
        for ep in range(100):
            m.train();opt.zero_grad();out=m(Xt);loss=lf(out,yt);loss.backward();opt.step()
            if loss.item()<bl:bl=loss.item();best={k:v.detach().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(best);m.eval()
        with torch.no_grad():proba[te]=m(torch.tensor(covs[te]).to(DEV)).cpu().numpy()
    return roc_auc_score(y,proba)
a=[run_spd(s) for s in SEEDS5]
res['SPDNet']=(np.mean(a),np.std(a)); print(f'  SPDNet     AUC={np.mean(a):.3f}±{np.std(a):.3f}',flush=True)

# --- VNN ---
torch.set_default_dtype(torch.float32)
COVn=np.array([c/np.trace(c)*NA for c in covs],dtype=np.float32)
class VNN(nn.Module):
    def __init__(s,n=NA,K=3,F1=4,F2=4):
        super().__init__();s.K=K
        s.h1=nn.Parameter(torch.randn(F1,K+1)*0.1);s.h2=nn.Parameter(torch.randn(F1,F2,K+1)*0.1)
        s.cls=nn.Sequential(nn.Linear(F2*n,16),nn.ReLU(),nn.Dropout(0.4),nn.Linear(16,1))
    def forward(s,Cmat,x):
        B=Cmat.shape[0];terms=[x];Ck=x
        for k in range(1,s.K+1):Ck=torch.einsum('bij,bj->bi',Cmat,Ck);terms.append(Ck)
        T=torch.stack(terms,1);h1=torch.relu(torch.einsum('fk,bkn->bfn',s.h1,T))
        terms2=[h1];Ck=h1
        for k in range(1,s.K+1):Ck=torch.einsum('bij,bfj->bfi',Cmat,Ck);terms2.append(Ck)
        T2=torch.stack(terms2,2);h2=torch.relu(torch.einsum('fgk,bfkn->bgn',s.h2,T2))
        return s.cls(h2.reshape(B,-1)).squeeze(-1)
def run_vnn(seed):
    torch.manual_seed(seed);np.random.seed(seed)
    skf=StratifiedKFold(5,shuffle=True,random_state=seed);proba=np.zeros(len(y))
    for tr,te in skf.split(COVn,y):
        m=VNN().to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=1e-2,weight_decay=1e-2)
        pw=torch.tensor([(y[tr]==0).sum()/max((y[tr]==1).sum(),1)],dtype=torch.float32).to(DEV)
        lf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Ct=torch.tensor(COVn[tr]).to(DEV);xt=torch.tensor(X0[tr]).to(DEV);yt=torch.tensor(y[tr],dtype=torch.float32).to(DEV)
        best=None;bl=1e9
        for ep in range(150):
            m.train();opt.zero_grad();out=m(Ct,xt);loss=lf(out,yt);loss.backward();opt.step()
            if loss.item()<bl:bl=loss.item();best={k:v.detach().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(best);m.eval()
        with torch.no_grad():proba[te]=m(torch.tensor(COVn[te]).to(DEV),torch.tensor(X0[te]).to(DEV)).cpu().numpy()
    return roc_auc_score(y,proba)
a=[run_vnn(s) for s in SEEDS]
res['VNN']=(np.mean(a),np.std(a)); print(f'  VNN        AUC={np.mean(a):.3f}±{np.std(a):.3f}',flush=True)

with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/model_daic.csv','w') as f:
    f.write('model,AUC,std\n')
    for k in ['tangent_LR','SPDNet','VNN','sparse_L1']:
        f.write(f'{k},{res[k][0]:.4f},{res[k][1]:.4f}\n')
print('\nDONE → model_daic.csv',flush=True)
