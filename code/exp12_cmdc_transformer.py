"""
exp12 — CMDC 앵커(Q3+Q7)에 Transformer 백본. baseline(Ridge 0.51)과 비교.
+ 앵커 有無 비교 (Transformer도 앵커 효과 유지되나).
AU 시퀀스 → Transformer encoder → PHQ 회귀. 지표 CCC/MAE/AUC.
소규모(n=44)라 작은 Transformer + 강한 정규화 + LOSO스러운 5fold.
"""
import numpy as np, warnings, openpyxl
from pathlib import Path
import torch, torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
warnings.filterwarnings('ignore')
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU_R=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r',
      'AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
NA=len(AU_R); T=96; SEEDS=[42,1,2,3,4]
def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean();vy,vp=y.var(),yp.var();cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0])
iID,iMDD=hd.index('ID'),hd.index('MDD'); iP=[hd.index(f'PHQ-{i}') for i in range(1,10)]
lab={}
for r in rows[1:]:
    if r[iID] is None: continue
    try: tot=sum(int(r[i]) for i in iP if r[i] is not None)
    except: continue
    lab[str(r[iID]).strip()]=(tot,int(r[iMDD]))
def au_q(subj,q):
    f=C/subj/f'Q{q}.csv'
    if not f.exists(): return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try: oi=h.index('success'); ai=[h.index(c) for c in AU_R]
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
def resample(x,T):
    if len(x)<2: return None
    idx=np.linspace(0,len(x)-1,T)
    return np.stack([np.interp(idx,np.arange(len(x)),x[:,j]) for j in range(x.shape[1])],1)
def seq_for(subj,qs):
    parts=[au_q(subj,q) for q in qs]
    if any(p is None for p in parts): return None
    cat=np.vstack(parts)
    return resample(cat,T)

class TFReg(nn.Module):
    def __init__(s,na=NA,d=48,heads=4,layers=2):
        super().__init__()
        s.inp=nn.Linear(na,d)
        s.pos=nn.Parameter(torch.randn(1,T,d)*0.02)
        el=nn.TransformerEncoderLayer(d,heads,d*2,dropout=0.3,batch_first=True)
        s.enc=nn.TransformerEncoder(el,layers)
        s.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.ReLU(),nn.Dropout(0.3),nn.Linear(d,1))
    def forward(s,x):
        h=s.inp(x)+s.pos; h=s.enc(h); h=h.mean(1)
        return s.head(h).squeeze(-1)

def run(qs,name):
    subs=[s for s in lab if seq_for(s,qs) is not None]
    X=np.array([seq_for(s,qs) for s in subs],dtype=np.float32)
    mu,sd_=X.reshape(-1,NA).mean(0),X.reshape(-1,NA).std(0)+1e-6
    X=(X-mu)/sd_
    ys=np.array([lab[s][0] for s in subs],dtype=np.float32); yb=np.array([lab[s][1] for s in subs])
    cccs,maes,aucs=[],[],[]
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        kf=KFold(5,shuffle=True,random_state=sd); pr=np.zeros(len(ys))
        for tr,te in kf.split(X):
            m=TFReg().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-2)
            ymu,ystd=ys[tr].mean(),ys[tr].std()+1e-6
            Xt=torch.tensor(X[tr]).to(DEV); yt=torch.tensor((ys[tr]-ymu)/ystd).to(DEV)
            for ep in range(120):
                m.train(); perm=torch.randperm(len(Xt))
                for i in range(0,len(Xt),8):
                    idx=perm[i:i+8]; opt.zero_grad()
                    loss=nn.functional.smooth_l1_loss(m(Xt[idx]),yt[idx]); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): p=m(torch.tensor(X[te]).to(DEV)).cpu().numpy()*ystd+ymu
            pr[te]=p
        cccs.append(ccc(ys,pr)); maes.append(mean_absolute_error(ys,pr))
        try: aucs.append(roc_auc_score(yb,pr))
        except: pass
    print(f'  [{name:22s}] n={len(subs)} CCC={np.mean(cccs):.3f}±{np.std(cccs):.3f} MAE={np.mean(maes):.2f} AUC={np.mean(aucs):.3f}',flush=True)

print('=== CMDC Transformer 백본 (baseline: Ridge 앵커 CCC0.51/AUC0.81) ===',flush=True)
run([3,7],'TF 앵커(Q3+Q7)')
run(list(range(1,13)),'TF 전체12질문')
