"""exp128 — 협응 vs 평균, 프레임 수 하한(MINF)별 비교. confound-free(오디오 유저내 표준화) 조건.
가설: 프레임이 충분하면 협응(SPD)이 평균을 역전한다. 안 하면 협응은 이 도메인에서 무효.
usage: exp128_minf.py <MINF> <coord|mean>
"""
import os, sys, pickle, random, numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
MINF=sys.argv[1]; VAR=sys.argv[2]
CACHE=os.path.expanduser(f'~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_minf{MINF}.pkl')
DEV='cuda'; D=20; MAXK=360; EPOCHS=40; BS=8; LR=1e-3
with open(CACHE,'rb') as f: C=pickle.load(f)
S={'train':[], 'val':[], 'test':[]}
for n,v in C.items():
    S[v['split']].append((v['covs'][:MAXK], v['mean'][:MAXK], v['aud'][:MAXK], v['label']))
ad=S['train'][0][2].shape[1]
M=np.vstack([m for _,m,_,_ in S['train']]); mmu,msd=M.mean(0), M.std(0)+1e-6

def pack(b):
    K=max(len(c) for c,_,_,_ in b); B=len(b)
    covs=np.zeros((B,K,D,D),np.float32); mean=np.zeros((B,K,D),np.float32)
    aud=np.zeros((B,K,ad),np.float32); m=np.zeros((B,K),np.float32); y=np.zeros(B,np.float32)
    for i,(c,mn,a,l) in enumerate(b):
        k=len(c); covs[i,:k]=c; covs[i,k:]=np.eye(D,dtype=np.float32)
        mean[i,:k]=(mn-mmu)/msd; m[i,:k]=1; y[i]=l
        aud[i,:k]=(a-a.mean(0))/(a.std(0)+1e-6)      # 유저내 표준화 = confound 제거
    t=torch.from_numpy; return t(covs),t(mean),t(aud),t(m),t(y)

def loader(sp,bs,sh):
    idx=list(range(len(S[sp])))
    if sh: random.shuffle(idx)
    for i in range(0,len(idx),bs): yield pack([S[sp][j] for j in idx[i:i+bs]])

class SPDseq(nn.Module):
    def __init__(s,d,out=16,emb=64):
        super().__init__(); s.W=nn.Parameter(torch.randn(out,d)*0.1); s.out=out
        s.pre=nn.Sequential(nn.Linear(out*(out+1)//2,emb), nn.ReLU())
    def sl(s,Mx):
        I=torch.eye(s.out,device=Mx.device,dtype=torch.float64)
        B=(s.W.double()@Mx.double()@s.W.double().T)+1e-2*I
        ev,U=torch.linalg.eigh(B); ev=ev.clamp(min=1e-3)
        M2=U@torch.diag_embed(ev)@U.transpose(1,2)
        e2,U2=torch.linalg.eigh(M2+1e-4*I)
        L=U2@torch.diag_embed(torch.log(e2.clamp(min=1e-4)))@U2.transpose(1,2)
        i=torch.triu_indices(s.out,s.out); return L[:,i[0],i[1]].float()
    def forward(s,c):
        B,K,d,_=c.shape; return s.pre(s.sl(c.reshape(B*K,d,d)).reshape(B,K,-1))

class Net(nn.Module):
    def __init__(s,h=48):
        super().__init__()
        s.enc=SPDseq(D) if VAR=='coord' else nn.Sequential(nn.Linear(D,64), nn.ReLU())
        s.ap=nn.Sequential(nn.Linear(ad,32), nn.ReLU())
        s.gru=nn.GRU(96,h,batch_first=True)
        s.head=nn.Sequential(nn.Dropout(0.3), nn.Linear(h,1))
    def forward(s,covs,mean,aud):
        z=s.enc(covs) if VAR=='coord' else s.enc(mean)
        z,_=s.gru(torch.cat([z,s.ap(aud)],-1))
        return s.head(z).squeeze(-1)

def ul(sl,m): return (sl*m).sum(1)/m.sum(1).clamp(min=1)
def ev(net,sp):
    net.eval(); U,Y=[],[]
    with torch.no_grad():
        for covs,mean,aud,m,y in loader(sp,BS,False):
            U+=ul(net(covs.to(DEV),mean.to(DEV),aud.to(DEV)),m.to(DEV)).cpu().tolist(); Y+=y.tolist()
    P=(np.array(U)>0).astype(int); Y=np.array(Y)
    return dict(f1=f1_score(Y,P,zero_division=0), acc=accuracy_score(Y,P), auc=roc_auc_score(Y,U))

npos=sum(x[3] for x in S['train']); nneg=len(S['train'])-npos
R=[]
for seed in range(5):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net=Net().to(DEV); opt=torch.optim.Adam(net.parameters(),LR,weight_decay=1e-4)
    pw=torch.tensor([nneg/max(1,npos)],device=DEV); bf,bsd=-1,None
    for e in range(EPOCHS):
        net.train()
        for covs,mean,aud,m,y in loader('train',BS,True):
            covs,mean,aud,m,y=covs.to(DEV),mean.to(DEV),aud.to(DEV),m.to(DEV),y.to(DEV)
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(ul(net(covs,mean,aud),m),y,pos_weight=pw).backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
        v=ev(net,'val')
        if v['f1']>=bf: bf=v['f1']; bsd={k:t.detach().clone() for k,t in net.state_dict().items()}
    net.load_state_dict(bsd); R.append(ev(net,'test'))
print(f"===== MINF={MINF} VAR={VAR} 5seed =====", flush=True)
for k in ['f1','acc','auc']:
    a=np.array([r[k] for r in R]); print(f'  {k} = {a.mean():.4f} ± {a.std():.4f}', flush=True)
print('DONE', flush=True)
