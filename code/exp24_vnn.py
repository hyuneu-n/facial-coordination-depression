"""
exp24 — coVariance Neural Network (VNN). 근거: Sihag et al. arXiv 2305.01807.
VNN = 공분산 C가 그래프인 GCN. 필터 H(C)=sum_k h_k C^k (학습형 PCA).
파라미터 극소 → 소규모 안전. 우리 pairwise coupling 발견과 원리 동일.
CMDC 앵커 Q3+Q7. baseline: tangent+logistic 0.885.
결과 CSV 저장.
"""
import numpy as np, warnings, openpyxl, csv
from pathlib import Path
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
NA=len(AU); SEEDS=list(range(10))
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
    return np.array(fe) if len(fe)>=10 else None
def seq(subj):
    parts=[c_q(subj,q) for q in [3,7]]
    if any(p is None for p in parts): return None
    return np.vstack(parts)
from sklearn.covariance import ledoit_wolf
subs=[s for s in CL if seq(s) is not None]
# 입력: 공분산 C(정규화) + 노드신호 x = AU 평균
COV=[]; X0=[]; y=[]
for s in subs:
    Xs=seq(s); c,_=ledoit_wolf(Xs)
    c=c/np.trace(c)*NA  # trace 정규화
    COV.append(c); X0.append(Xs.mean(0)); y.append(CL[s])
COV=np.array(COV,dtype=np.float32); X0=np.array(X0,dtype=np.float32); y=np.array(y)
print(f'VNN 입력: n={len(y)} cov{COV.shape} MDD{int(y.sum())}/HC{int((y==0).sum())}',flush=True)

class VNN(nn.Module):
    """coVariance filter: H(C)x = sum_{k=0}^K h_k C^k x. 2 layer + 다중필터."""
    def __init__(s,n=NA,K=3,F1=4,F2=4):
        super().__init__()
        s.K=K
        s.h1=nn.Parameter(torch.randn(F1,K+1)*0.1)   # layer1 필터계수
        s.h2=nn.Parameter(torch.randn(F1,F2,K+1)*0.1)
        s.cls=nn.Sequential(nn.Linear(F2*n,16),nn.ReLU(),nn.Dropout(0.4),nn.Linear(16,1))
    def cov_filter(s,Cmat,x,h):
        # Cmat:(B,n,n), x:(B,n) or (B,Fin,n); h: (...,K+1)
        out=0; Ck=x
        terms=[x]
        for k in range(1,s.K+1):
            Ck=torch.einsum('bij,bfj->bfi',Cmat,Ck) if x.dim()==3 else torch.einsum('bij,bj->bi',Cmat,Ck)
            terms.append(Ck)
        return terms
    def forward(s,Cmat,x):
        B=Cmat.shape[0]
        # layer1: x(B,n) -> (B,F1,n)
        terms=[x]; Ck=x
        for k in range(1,s.K+1):
            Ck=torch.einsum('bij,bj->bi',Cmat,Ck); terms.append(Ck)
        T=torch.stack(terms,1)  # (B,K+1,n)
        h1=torch.relu(torch.einsum('fk,bkn->bfn',s.h1,T))  # (B,F1,n)
        # layer2: (B,F1,n)->(B,F2,n)
        terms2=[h1]; Ck=h1
        for k in range(1,s.K+1):
            Ck=torch.einsum('bij,bfj->bfi',Cmat,Ck); terms2.append(Ck)
        T2=torch.stack(terms2,2)  # (B,F1,K+1,n)
        h2=torch.relu(torch.einsum('fgk,bfkn->bgn',s.h2,T2))  # (B,F2,n)
        return s.cls(h2.reshape(B,-1)).squeeze(-1)

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); proba=np.zeros(len(y))
    for tr,te in skf.split(COV,y):
        m=VNN().to(DEV)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-2,weight_decay=1e-2)
        pw=torch.tensor([(y[tr]==0).sum()/max((y[tr]==1).sum(),1)],dtype=torch.float32).to(DEV)
        lf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Ct=torch.tensor(COV[tr]).to(DEV); xt=torch.tensor(X0[tr]).to(DEV); yt=torch.tensor(y[tr],dtype=torch.float32).to(DEV)
        best=None;bl=1e9
        for ep in range(150):
            m.train(); opt.zero_grad(); out=m(Ct,xt); loss=lf(out,yt); loss.backward(); opt.step()
            if loss.item()<bl: bl=loss.item(); best={k:v.detach().clone() for k,v in m.state_dict().items()}
        m.load_state_dict(best); m.eval()
        with torch.no_grad(): proba[te]=m(torch.tensor(COV[te]).to(DEV),torch.tensor(X0[te]).to(DEV)).cpu().numpy()
    return roc_auc_score(y,proba)

aucs=[run(s) for s in SEEDS]
res=f'VNN AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}'
print(res,flush=True)
print(f'(baseline tangent+logistic: 0.885)',flush=True)
verdict='VNN 우위/동등 ✅' if np.mean(aucs)>=0.87 else ('경쟁력 있음' if np.mean(aucs)>=0.80 else '열세')
print(f'→ {verdict}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/vnn_result.csv','w') as f:
    f.write('model,AUC,std,baseline\n')
    f.write(f'VNN,{np.mean(aucs):.4f},{np.std(aucs):.4f},0.885\n')
