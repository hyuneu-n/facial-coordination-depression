"""exp134 — Procrustes 정렬(위치·크기·회전 제거) 표현으로 학습. MUD3의 정직한 상한.
usage: exp134_aligned.py <face|face_aud>
"""
import os, sys, pickle, random, numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
MODE=sys.argv[1] if len(sys.argv)>1 else 'face'
CACHE=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_aligned.pkl')
DEV='cuda'; D=20; MAXK=360; EPOCHS=40; BS=8; LR=1e-3
C=pickle.load(open(CACHE,'rb'))
S={'train':[], 'val':[], 'test':[]}
for n,v in C.items(): S[v['split']].append((v['mean'][:MAXK], v['aud'][:MAXK], v['label']))
ad=S['train'][0][1].shape[1]
M=np.vstack([m for m,_,_ in S['train']]); mmu,msd=M.mean(0), M.std(0)+1e-6
print(f'{MODE}: ' + str({k:(len(v),sum(x[2] for x in v)) for k,v in S.items()}), flush=True)

def pack(b):
    K=max(len(m) for m,_,_ in b); B=len(b)
    mean=np.zeros((B,K,D),np.float32); aud=np.zeros((B,K,ad),np.float32)
    msk=np.zeros((B,K),np.float32); y=np.zeros(B,np.float32)
    for i,(mn,a,l) in enumerate(b):
        k=len(mn); mean[i,:k]=(mn-mmu)/msd; msk[i,:k]=1; y[i]=l
        sd0=a.std(0); aud[i,:k]=(a-a.mean(0))/np.where(sd0<1e-8,1.0,sd0)
    t=torch.from_numpy; return t(mean),t(aud),t(msk),t(y)

def loader(sp,bs,sh):
    idx=list(range(len(S[sp])))
    if sh: random.shuffle(idx)
    for i in range(0,len(idx),bs): yield pack([S[sp][j] for j in idx[i:i+bs]])

class Net(nn.Module):
    def __init__(s,h=48):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(D,64), nn.ReLU())
        s.use_a = MODE=='face_aud'
        s.ap=nn.Sequential(nn.Linear(ad,32), nn.ReLU()) if s.use_a else None
        s.gru=nn.GRU(64+(32 if s.use_a else 0),h,batch_first=True)
        s.head=nn.Sequential(nn.Dropout(0.3), nn.Linear(h,1))
    def forward(s,mean,aud):
        z=s.enc(mean)
        if s.use_a: z=torch.cat([z,s.ap(aud)],-1)
        z,_=s.gru(z); return s.head(z).squeeze(-1)

def ul(sl,m): return (sl*m).sum(1)/m.sum(1).clamp(min=1)
def ev(net,sp):
    net.eval(); U,Y=[],[]
    with torch.no_grad():
        for mean,aud,m,y in loader(sp,BS,False):
            U+=ul(net(mean.to(DEV),aud.to(DEV)),m.to(DEV)).cpu().tolist(); Y+=y.tolist()
    P=(np.array(U)>0).astype(int); Y=np.array(Y)
    return dict(f1=f1_score(Y,P,zero_division=0), acc=accuracy_score(Y,P), auc=roc_auc_score(Y,U))

npos=sum(x[2] for x in S['train']); nneg=len(S['train'])-npos
R=[]
for seed in range(5):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net=Net().to(DEV)
    if seed==0: print(f'params={sum(p.numel() for p in net.parameters()):,}', flush=True)
    opt=torch.optim.Adam(net.parameters(),LR,weight_decay=1e-4)
    pw=torch.tensor([nneg/max(1,npos)],device=DEV); bf,bsd=-1,None
    for e in range(EPOCHS):
        net.train()
        for mean,aud,m,y in loader('train',BS,True):
            mean,aud,m,y=mean.to(DEV),aud.to(DEV),m.to(DEV),y.to(DEV)
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(ul(net(mean,aud),m),y,pos_weight=pw).backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
        v=ev(net,'val')
        if v['f1']>=bf: bf=v['f1']; bsd={k:t.detach().clone() for k,t in net.state_dict().items()}
    net.load_state_dict(bsd); t=ev(net,'test'); R.append(t)
    print(f">> [{MODE}] seed{seed} f1={t['f1']:.4f} auc={t['auc']:.4f}", flush=True)
print(f"\n===== exp134 정렬표현 [{MODE}] 5seed =====", flush=True)
for k in ['f1','acc','auc']:
    a=np.array([r[k] for r in R]); print(f'  {k} = {a.mean():.4f} ± {a.std():.4f}', flush=True)
print('DONE', flush=True)
