"""exp135 — 진짜 계층 모델. baseline이 코드상 붕괴시킨 2단 구조를 제대로 구현.
Level1: 영상 내 프레임 시퀀스 인코더 (길이 mask 적용)
Level2: 영상 시퀀스 GRU (시간순) + mask pooling
usage: exp135_hier.py <hier|flat> <MAXK: 360|0(전량)>
"""
import os, sys, pickle, random, time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
MODE=sys.argv[1] if len(sys.argv)>1 else 'hier'
MAXK=int(sys.argv[2]) if len(sys.argv)>2 else 360
CACHE=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_seq.pkl')
DEV='cuda'; DV=20; DA=25; TMAX=60; EPOCHS=30; BS=4; LR=1e-3
C=pickle.load(open(CACHE,'rb'))
S={'train':[], 'val':[], 'test':[]}
for n,v in C.items():
    k=len(v['vis']) if MAXK==0 else min(len(v['vis']),MAXK)
    S[v['split']].append((v['vis'][:k], v['aud'][:k], v['lens'][:k], v['label']))
del C
print(f'{MODE} MAXK={MAXK} ' + str({k:(len(v),sum(x[3] for x in v)) for k,v in S.items()}), flush=True)

def pack(b):
    K=max(len(x[0]) for x in b); B=len(b)
    vis=np.zeros((B,K,TMAX,DV),np.float32); aud=np.zeros((B,K,TMAX,DA),np.float32)
    fm=np.zeros((B,K,TMAX),np.float32); vm=np.zeros((B,K),np.float32); y=np.zeros(B,np.float32)
    for i,(v,a,l,lab) in enumerate(b):
        k=len(v); vis[i,:k]=v; aud[i,:k]=a; vm[i,:k]=1; y[i]=lab
        for j,t in enumerate(l): fm[i,j,:int(t)]=1        # 프레임 유효 길이 mask
    t=torch.from_numpy; return t(vis),t(aud),t(fm),t(vm),t(y)

def loader(sp,bs,sh):
    idx=list(range(len(S[sp])))
    if sh: random.shuffle(idx)
    for i in range(0,len(idx),bs): yield pack([S[sp][j] for j in idx[i:i+bs]])

class VideoEnc(nn.Module):
    """Level1 — 영상 내 프레임 시계열 인코더"""
    def __init__(s, cin, d=64):
        super().__init__()
        s.conv=nn.Sequential(nn.Conv1d(cin,48,5,padding=2), nn.BatchNorm1d(48), nn.ReLU(),
                             nn.Conv1d(48,d,3,padding=1), nn.BatchNorm1d(d), nn.ReLU())
    def forward(s,x,fm):
        # x [N,T,C] fm [N,T]
        h=s.conv(x.transpose(1,2)).transpose(1,2)          # [N,T,d]
        w=fm.unsqueeze(-1)
        return (h*w).sum(1)/w.sum(1).clamp(min=1)          # masked mean → [N,d]

class Net(nn.Module):
    def __init__(s,h=48,d=64):
        super().__init__()
        s.hier = MODE=='hier'
        if s.hier: s.enc=VideoEnc(DV+DA,d)
        else:      s.enc=nn.Sequential(nn.Linear(DV+DA,d), nn.ReLU())
        s.gru=nn.GRU(d,h,batch_first=True)
        s.head=nn.Sequential(nn.Dropout(0.3), nn.Linear(h,1))
    def forward(s,vis,aud,fm,vm):
        B,K,T,_=vis.shape
        x=torch.cat([vis,aud],-1)
        if s.hier:
            z=s.enc(x.reshape(B*K,T,-1), fm.reshape(B*K,T)).reshape(B,K,-1)
        else:
            w=fm.unsqueeze(-1)
            z=s.enc((x*w).sum(2)/w.sum(2).clamp(min=1))    # 영상 내 평균 후 인코딩 (기존 방식)
        z,_=s.gru(z)
        return s.head(z).squeeze(-1)

def ul(sl,vm): return (sl*vm).sum(1)/vm.sum(1).clamp(min=1)
def ev(net,sp):
    net.eval(); U,Y=[],[]
    with torch.no_grad():
        for vis,aud,fm,vm,y in loader(sp,BS,False):
            U+=ul(net(vis.to(DEV),aud.to(DEV),fm.to(DEV),vm.to(DEV)),vm.to(DEV)).cpu().tolist(); Y+=y.tolist()
    P=(np.array(U)>0).astype(int); Y=np.array(Y)
    return dict(f1=f1_score(Y,P,zero_division=0), acc=accuracy_score(Y,P), auc=roc_auc_score(Y,U))

npos=sum(x[3] for x in S['train']); nneg=len(S['train'])-npos
R=[]
for seed in range(3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net=Net().to(DEV)
    if seed==0: print(f'params={sum(p.numel() for p in net.parameters()):,}', flush=True)
    opt=torch.optim.Adam(net.parameters(),LR,weight_decay=1e-4)
    pw=torch.tensor([nneg/max(1,npos)],device=DEV); bf,bsd=-1,None
    for e in range(EPOCHS):
        net.train(); t0=time.time()
        for vis,aud,fm,vm,y in loader('train',BS,True):
            vis,aud,fm,vm,y=vis.to(DEV),aud.to(DEV),fm.to(DEV),vm.to(DEV),y.to(DEV)
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(ul(net(vis,aud,fm,vm),vm),y,pos_weight=pw).backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
        v=ev(net,'val')
        if v['f1']>=bf: bf=v['f1']; bsd={k:t.detach().clone() for k,t in net.state_dict().items()}
        if e%10==0: print(f'  seed{seed} ep{e} val f1={v["f1"]:.4f} ({time.time()-t0:.0f}s)', flush=True)
    net.load_state_dict(bsd); t=ev(net,'test'); R.append(t)
    print(f">> [{MODE} MAXK={MAXK}] seed{seed} f1={t['f1']:.4f} auc={t['auc']:.4f}", flush=True)
print(f"\n===== exp135 [{MODE}] MAXK={MAXK} 3seed =====", flush=True)
for k in ['f1','acc','auc']:
    a=np.array([r[k] for r in R]); print(f'  {k} = {a.mean():.4f} ± {a.std():.4f}', flush=True)
print('DONE', flush=True)
