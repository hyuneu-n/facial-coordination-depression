"""exp125 — MUD3 계층협응 ablation. 협응(SPD)이 실제로 기여하는지 검증.
variants: full / no_coord(프레임평균) / no_gru(pooling) / no_audio / no_mask / no_shrink(경험공분산)
"""
import os, sys, pickle, random, time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
VAR = sys.argv[1] if len(sys.argv)>1 else 'full'
CACHE = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_d20.pkl')
MEANC = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_mean20.pkl')  # no_coord용
DEV='cuda'; D=20; MAXK=360; EPOCHS=40; BS=8; LR=1e-3

with open(CACHE,'rb') as f: C=pickle.load(f)
MEAN=None
if VAR=='no_coord':
    with open(MEANC,'rb') as f: MEAN=pickle.load(f)
S={'train':[], 'val':[], 'test':[]}
for name,v in C.items():
    extra = MEAN[name][:MAXK] if MEAN is not None else None
    S[v['split']].append((name, v['covs'][:MAXK], v['aud'][:MAXK], v['label'], extra))
A=np.vstack([a for _,_,a,_,_ in S['train']]); amu,asd=A.mean(0), A.std(0)+1e-6
if VAR=='no_coord':
    M=np.vstack([e for _,_,_,_,e in S['train']]); mmu,msd=M.mean(0), M.std(0)+1e-6

def pack(b):
    K=max(len(c) for _,c,_,_,_ in b); B=len(b)
    covs=np.zeros((B,K,D,D),np.float32); aud=np.zeros((B,K,len(amu)),np.float32)
    ext=np.zeros((B,K,D),np.float32) if VAR=='no_coord' else None
    m=np.zeros((B,K),np.float32); y=np.zeros(B,np.float32)
    for i,(n,c,a,l,e) in enumerate(b):
        k=len(c); covs[i,:k]=c; aud[i,:k]=(a-amu)/asd; m[i,:k]=1; y[i]=l
        covs[i,k:]=np.eye(D,dtype=np.float32)
        if ext is not None: ext[i,:k]=(e-mmu)/msd
    t=lambda x: torch.from_numpy(x)
    return t(covs), t(aud), t(m), t(y), (t(ext) if ext is not None else None)

def loader(sp,bs,sh):
    idx=list(range(len(S[sp])))
    if sh: random.shuffle(idx)
    for i in range(0,len(idx),bs): yield pack([S[sp][j] for j in idx[i:i+bs]])

class SPDseq(nn.Module):
    def __init__(s,d,out=16,emb=64):
        super().__init__(); s.W=nn.Parameter(torch.randn(out,d)*0.1); s.out=out
        s.pre=nn.Sequential(nn.Linear(out*(out+1)//2,emb), nn.ReLU())
    def sl(s,M):
        I=torch.eye(s.out,device=M.device,dtype=torch.float64)
        B=(s.W.double()@M.double()@s.W.double().T)+1e-2*I
        ev,U=torch.linalg.eigh(B); ev=ev.clamp(min=1e-3)
        M2=U@torch.diag_embed(ev)@U.transpose(1,2)
        e2,U2=torch.linalg.eigh(M2+1e-4*I)
        L=U2@torch.diag_embed(torch.log(e2.clamp(min=1e-4)))@U2.transpose(1,2)
        i=torch.triu_indices(s.out,s.out); return L[:,i[0],i[1]].float()
    def forward(s,c):
        B,K,d,_=c.shape; return s.pre(s.sl(c.reshape(B*K,d,d)).reshape(B,K,-1))

class Net(nn.Module):
    def __init__(s,ad,h=48):
        super().__init__()
        s.var=VAR
        s.enc = nn.Sequential(nn.Linear(D,64), nn.ReLU()) if VAR=='no_coord' else SPDseq(D)
        s.use_a = VAR!='no_audio'
        s.ap=nn.Sequential(nn.Linear(ad,32), nn.ReLU()) if s.use_a else None
        ind=64+(32 if s.use_a else 0)
        s.gru=None if VAR=='no_gru' else nn.GRU(ind,h,batch_first=True)
        s.head=nn.Sequential(nn.Dropout(0.3), nn.Linear(h if VAR!='no_gru' else ind, 1))
    def forward(s,covs,aud,ext):
        z = s.enc(ext) if s.var=='no_coord' else s.enc(covs)
        if s.use_a: z=torch.cat([z, s.ap(aud)],-1)
        if s.gru is not None: z,_=s.gru(z)
        return s.head(z).squeeze(-1)

def ulogit(sl,m):
    if VAR=='no_mask': return sl.mean(1)
    return (sl*m).sum(1)/m.sum(1).clamp(min=1)

def ev(net,sp):
    net.eval(); UL,Y=[],[]
    with torch.no_grad():
        for covs,aud,m,y,ext in loader(sp,BS,False):
            sl=net(covs.to(DEV),aud.to(DEV),ext.to(DEV) if ext is not None else None)
            UL+=ulogit(sl,m.to(DEV)).cpu().tolist(); Y+=y.tolist()
    P=(np.array(UL)>0).astype(int); Y=np.array(Y)
    return dict(f1=f1_score(Y,P,zero_division=0), acc=accuracy_score(Y,P), auc=roc_auc_score(Y,UL))

npos=sum(x[3] for x in S['train']); nneg=len(S['train'])-npos
R=[]
for seed in range(5):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net=Net(len(amu)).to(DEV)
    if seed==0: print(f'[{VAR}] params={sum(p.numel() for p in net.parameters()):,}', flush=True)
    opt=torch.optim.Adam(net.parameters(),LR,weight_decay=1e-4)
    pw=torch.tensor([nneg/max(1,npos)],device=DEV); bf,bs_=-1,None
    for e in range(EPOCHS):
        net.train()
        for covs,aud,m,y,ext in loader('train',BS,True):
            covs,aud,m,y=covs.to(DEV),aud.to(DEV),m.to(DEV),y.to(DEV)
            ext=ext.to(DEV) if ext is not None else None
            opt.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(ulogit(net(covs,aud,ext),m),y,pos_weight=pw)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
        v=ev(net,'val')
        if v['f1']>=bf: bf=v['f1']; bs_={k:t.detach().clone() for k,t in net.state_dict().items()}
    net.load_state_dict(bs_); t=ev(net,'test')
    print(f">> [{VAR}] seed{seed} f1={t['f1']:.4f} acc={t['acc']:.4f} auc={t['auc']:.4f}", flush=True)
    R.append(t)
print(f"\n===== ablation [{VAR}] 5seed =====", flush=True)
for k in ['f1','acc','auc']:
    a=np.array([r[k] for r in R]); print(f'  {k} = {a.mean():.4f} ± {a.std():.4f}', flush=True)
print('DONE', flush=True)
