"""
exp21 — SPDNet (공분산 위 딥러닝) vs tangent-LR baseline (0.885).
SPDNet 핵심층 직접 구현: BiMap(Stiefel) -> ReEig -> LogEig -> vec -> FC.
CMDC 앵커(Q3+Q7) AU 공분산. 소규모라 1 BiMap block + 강한 정규화.
"이기면 deep geometry pays off = novelty".
"""
import numpy as np, warnings, openpyxl
from pathlib import Path
import torch, torch.nn as nn
import geoopt
from sklearn.covariance import ledoit_wolf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
NA=len(AU); SEEDS=[42,1,2,3,4]

wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0]); iID,iMDD=hd.index('ID'),hd.index('MDD')
CL={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def c_q(subj,q):
    f=C/subj/f'Q{q}.csv'
    if not f.exists(): return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try: oi=h.index('success'); ai=[h.index(c) for c in AU]
    except: return None
    fe=[]
    with open(f) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(fe) if len(fe)>=5 else None
def seq(subj,qs):
    parts=[c_q(subj,q) for q in qs]
    if any(p is None for p in parts): return None
    return np.vstack(parts)
def lw(X):
    c,_=ledoit_wolf(X); return c
subs=[s for s in CL if seq(s,[3,7]) is not None]
covs=np.array([lw(seq(s,[3,7])) for s in subs]); y=np.array([CL[s] for s in subs])
print(f'CMDC 앵커 Q3+Q7: n={len(y)}, MDD{int(y.sum())}/HC{int((y==0).sum())}, cov {covs.shape}\n',flush=True)

# ===== SPDNet 층 =====
class BiMap(nn.Module):
    """SPD -> SPD: W X W^T, W는 Stiefel(직교) 위에서 학습. n_in->n_out (n_out<=n_in)"""
    def __init__(s,n_in,n_out):
        super().__init__()
        w=torch.empty(n_in,n_out); nn.init.orthogonal_(w)
        s.W=geoopt.ManifoldParameter(w, manifold=geoopt.Stiefel())
    def forward(s,X):  # X:(B,n_in,n_in)
        W=s.W  # (n_in,n_out)
        return torch.einsum('ji,bjk,kl->bil', W, X, W)  # (B,n_out,n_out)
class ReEig(nn.Module):
    """eigenvalue rectification (SPD 유지 비선형)"""
    def __init__(s,eps=1e-4): super().__init__(); s.eps=eps
    def forward(s,X):
        X=(X+X.transpose(-1,-2))/2
        val,vec=torch.linalg.eigh(X)
        val=torch.clamp(val,min=s.eps)
        return vec@torch.diag_embed(val)@vec.transpose(-1,-2)
class LogEig(nn.Module):
    """SPD -> tangent(대칭행렬), matrix log"""
    def forward(s,X):
        X=(X+X.transpose(-1,-2))/2
        val,vec=torch.linalg.eigh(X)
        val=torch.clamp(val,min=1e-6)
        return vec@torch.diag_embed(torch.log(val))@vec.transpose(-1,-2)
class SPDNet(nn.Module):
    def __init__(s,n=NA,h=10):
        super().__init__()
        s.bimap=BiMap(n,h); s.reeig=ReEig(); s.logeig=LogEig()
        s.bn=nn.BatchNorm1d(h*h)
        s.fc=nn.Sequential(nn.Dropout(0.4), nn.Linear(h*h,1))
    def forward(s,X):
        X=s.reeig(s.bimap(X)); X=s.logeig(X)
        v=X.reshape(X.shape[0],-1)
        v=s.bn(v)
        return s.fc(v).squeeze(-1)

def run_spdnet(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); proba=np.zeros(len(y))
    for tr,te in skf.split(covs,y):
        m=SPDNet().to(DEV)
        opt=geoopt.optim.RiemannianAdam(m.parameters(),lr=5e-3,weight_decay=1e-3)
        pw=torch.tensor([(y[tr]==0).sum()/max((y[tr]==1).sum(),1)],dtype=torch.float64).to(DEV)
        lf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt=torch.tensor(covs[tr]).to(DEV); yt=torch.tensor(y[tr],dtype=torch.float64).to(DEV)
        best=None;bestl=1e9
        for ep in range(100):
            m.train(); opt.zero_grad(); out=m(Xt); loss=lf(out,yt); loss.backward(); opt.step()
            if loss.item()<bestl: bestl=loss.item(); best={k:v.detach().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(best); m.eval()
        with torch.no_grad(): proba[te]=m(torch.tensor(covs[te]).to(DEV)).cpu().numpy()
    return roc_auc_score(y,proba)

print('=== SPDNet (공분산 딥러닝) ===',flush=True)
aucs=[run_spdnet(s) for s in SEEDS]
print(f'SPDNet AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)
print(f'(baseline tangent-LR: 0.885)',flush=True)
print(f'→ {"deep geometry 이득 ✅" if np.mean(aucs)>0.885 else "tangent-LR이 나음(soga)"}',flush=True)
