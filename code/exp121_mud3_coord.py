"""exp121 — MUD3 계층적 협응 이식.
영상 = 프레임협응 SPD (ledoit_wolf, exp80 nf정규화+PCA20) → SPDNet
유저 = 영상 시퀀스 GRU (단방향=조기탐지 정당, 양방향=분류비교)
baseline(LSTM_han) F1과 조기탐지(ERDE/F1_latency) 직접 비교.
"""
import os, sys, math, pickle, random, time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
sys.path.insert(0, os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/CCAC-baseline'))
from utils import f_latency

CACHE = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_d20.pkl')
DEV='cuda'; D=20; MAXK=360; EPOCHS=40; BS=8; LR=1e-3; ALPHA=30.0
BIDIR = (len(sys.argv)>1 and sys.argv[1]=='bi')

with open(CACHE,'rb') as f: C = pickle.load(f)
S = {'train':[], 'val':[], 'test':[]}
for name,v in C.items(): S[v['split']].append((name, v['covs'][:MAXK], v['aud'][:MAXK], v['label']))
print({k:(len(v), sum(x[3] for x in v)) for k,v in S.items()}, flush=True)

# acoustic 표준화 (train 통계만)
A = np.vstack([a for _,_,a,_ in S['train']]); amu, asd = A.mean(0), A.std(0)+1e-6
print(f'aud dim={A.shape[1]}', flush=True)

def pack(batch):
    K = max(len(c) for _,c,_,_ in batch); B=len(batch)
    covs = np.zeros((B,K,D,D), np.float32); aud = np.zeros((B,K,len(amu)), np.float32)
    m = np.zeros((B,K), np.float32); y = np.zeros(B, np.float32)
    for i,(n,c,a,l) in enumerate(batch):
        k=len(c); covs[i,:k]=c; aud[i,:k]=(a-amu)/asd; m[i,:k]=1; y[i]=l
        covs[i,k:] = np.eye(D, dtype=np.float32)          # 패딩은 항등행렬(eigh 안전)
    return (torch.from_numpy(covs), torch.from_numpy(aud), torch.from_numpy(m), torch.from_numpy(y))

def loader(split, bs, shuffle):
    idx=list(range(len(S[split])))
    if shuffle: random.shuffle(idx)
    for i in range(0,len(idx),bs): yield pack([S[split][j] for j in idx[i:i+bs]])

class SPDseq(nn.Module):
    """exp110 SPDseq와 동일: BiMap → log-eig → tangent triu → Linear"""
    def __init__(self, d, out=16, emb=64):
        super().__init__(); self.W=nn.Parameter(torch.randn(out,d)*0.1); self.out=out
        self.pre=nn.Sequential(nn.Linear(out*(out+1)//2, emb), nn.ReLU())
    def sl(self, M):
        I=torch.eye(self.out, device=M.device, dtype=torch.float64)
        B=(self.W.double()@M.double()@self.W.double().T)+1e-2*I
        ev,U=torch.linalg.eigh(B); ev=ev.clamp(min=1e-3)
        M2=U@torch.diag_embed(ev)@U.transpose(1,2)
        ev2,U2=torch.linalg.eigh(M2+1e-4*I)
        L=U2@torch.diag_embed(torch.log(ev2.clamp(min=1e-4)))@U2.transpose(1,2)
        idx=torch.triu_indices(self.out,self.out)
        return L[:,idx[0],idx[1]].float()
    def forward(self, covs):
        B,K,d,_=covs.shape
        return self.pre(self.sl(covs.reshape(B*K,d,d)).reshape(B,K,-1))

class HierCoord(nn.Module):
    def __init__(self, d, ad, h=48, bidir=False):
        super().__init__()
        self.spd=SPDseq(d); self.ap=nn.Sequential(nn.Linear(ad,32), nn.ReLU())
        self.gru=nn.GRU(64+32, h, batch_first=True, bidirectional=bidir)
        self.head=nn.Sequential(nn.Dropout(0.3), nn.Linear(h*(2 if bidir else 1), 1))
    def forward(self, covs, aud):
        z=torch.cat([self.spd(covs), self.ap(aud)], -1)
        o,_=self.gru(z)
        return self.head(o).squeeze(-1)        # [B,K] step별 logit

def masked_user_logit(step_logits, m):
    return (step_logits*m).sum(1)/m.sum(1).clamp(min=1)

def early_loss(step_logits, y, m, alpha=ALPHA):
    """baseline EarlyDetectionLoss와 동일 형태: 늦은 탐지에 가중 페널티"""
    B,K=step_logits.shape
    bce=nn.functional.binary_cross_entropy_with_logits(step_logits, y[:,None].expand(-1,K), reduction='none')
    w=torch.ones_like(bce)
    ramp=torch.linspace(0, alpha, K, device=step_logits.device)[None,:].expand(B,-1)
    w=w+ramp*y[:,None]
    return (bce*w*m).sum()/m.sum().clamp(min=1)

def lc0k(target, traj, o):
    for i,t in enumerate(traj):
        if t==1:
            return 1*(1-1/(1+math.exp(i-o))) if target==1 else 0.4545
    return 0 if target==0 else 1

def evaluate(net, split, early=False):
    net.eval(); UL,Y,STEP,MM=[],[],[],[]
    with torch.no_grad():
        for covs,aud,m,y in loader(split,BS,False):
            sl=net(covs.to(DEV), aud.to(DEV)); mm=m.to(DEV)
            UL+=masked_user_logit(sl,mm).cpu().tolist(); Y+=y.tolist()
            if early: STEP.append((torch.sigmoid(sl)>0.5).long().cpu()*m.long()); MM.append(m)
    P=(np.array(UL)>0).astype(int); Y=np.array(Y)
    out=dict(f1=f1_score(Y,P,zero_division=0), acc=accuracy_score(Y,P),
             p=precision_score(Y,P,zero_division=0), r=recall_score(Y,P,zero_division=0),
             auc=roc_auc_score(Y,UL) if len(set(Y))>1 else float('nan'))
    if early:
        e5,e10,delays,epred=[],[],[],[]
        for sb,mb in zip(STEP,MM):
            for i in range(sb.shape[0]):
                L=int(mb[i].sum()); traj=sb[i,:L].tolist()
                t=Y[len(delays)]
                e5.append(lc0k(t,traj,5)); e10.append(lc0k(t,traj,10))
                fi=next((j for j,v in enumerate(traj) if v==1), None)
                epred.append(1 if fi is not None else 0)
                delays.append(fi if (t==1 and fi is not None) else (0 if t==0 else L))
        epred=np.array(epred)
        out.update(early_f1=f1_score(Y,epred,zero_division=0), early_acc=accuracy_score(Y,epred),
                   erde5=float(np.mean(e5)), erde10=float(np.mean(e10)),
                   f1_lat=float(f_latency(epred,Y,np.array(delays))))
    return out

npos=sum(x[3] for x in S['train']); nneg=len(S['train'])-npos
res=[]
for seed in [0,1,2,3,4,5,6,7,8,9]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net=HierCoord(D, len(amu), bidir=BIDIR).to(DEV)
    if seed==0: print(f'params={sum(p.numel() for p in net.parameters()):,}  bidir={BIDIR}', flush=True)
    opt=torch.optim.Adam(net.parameters(), LR, weight_decay=1e-4)
    pw=torch.tensor([nneg/max(1,npos)], device=DEV)
    bestf1, bestsd = -1, None
    for ep in range(EPOCHS):
        net.train(); t0=time.time()
        for covs,aud,m,y in loader('train',BS,True):
            covs,aud,m,y=covs.to(DEV),aud.to(DEV),m.to(DEV),y.to(DEV)
            opt.zero_grad()
            sl=net(covs,aud)
            ul=masked_user_logit(sl,m)
            loss=nn.functional.binary_cross_entropy_with_logits(ul,y,pos_weight=pw)+0.3*early_loss(sl,y,m)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
        v=evaluate(net,'val')
        if v['f1']>=bestf1: bestf1=v['f1']; bestsd={k:t.detach().clone() for k,t in net.state_dict().items()}
        if ep%5==0 or ep==EPOCHS-1: print(f"  seed{seed} ep{ep} val f1={v['f1']:.4f} acc={v['acc']:.4f} auc={v['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    net.load_state_dict(bestsd)
    t=evaluate(net,'test',early=True)
    print(f">> seed{seed} TEST f1={t['f1']:.4f} acc={t['acc']:.4f} auc={t['auc']:.4f} | early_f1={t['early_f1']:.4f} early_acc={t['early_acc']:.4f} ERDE5={t['erde5']:.4f} ERDE10={t['erde10']:.4f} F1lat={t['f1_lat']:.4f}", flush=True)
    res.append(t)

print(f"\n===== exp121 MUD3 계층협응 ({'bi' if BIDIR else 'uni'}-GRU, 3seed) =====", flush=True)
for k in ['f1','acc','auc','early_f1','early_acc','erde5','erde10','f1_lat']:
    a=np.array([r[k] for r in res]); print(f'  {k:10s} = {a.mean():.4f} ± {a.std():.4f}', flush=True)
print('  [baseline LSTM_han seed110] test F1=0.6786 acc=0.7273 | early_f1=0.6364 early_acc=0.5152 ERDE5=0.3307 ERDE10=0.2892 F1lat=0.6246', flush=True)
print('DONE', flush=True)
