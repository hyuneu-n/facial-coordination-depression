"""exp123 — 결정적 실험: baseline(LSTM_han)에 nf() 정규화 입력을 주면 성능이 떨어지는가?
떨어지면 = baseline 성능이 '얼굴 위치·크기' leakage에 의존했다는 직접 증거.
모델·하이퍼파라미터는 repro_baseline.py와 완전 동일. 입력 landmark 정규화 여부만 다름.
"""
import sys, os, pickle, random, time, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/CCAC-baseline'))
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from models import LSTM_han

R = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
DEV='cuda'; EPOCHS=20; BS=2; LR=1e-5
NORM = sys.argv[1] if len(sys.argv)>1 else 'norm'      # 'norm' | 'raw'

def nf(V):
    T=V.shape[0]; P=V.reshape(T,68,2).astype(np.float64)
    P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6
    return (P/sc[:,None]).reshape(T,136)

lab=pd.read_csv(f'{R}/labels.csv'); split_of=dict(zip(lab['names'],lab['split']))
data={'train':[], 'val':[], 'test':[]}
for fn,y in [('dep_feat.pkl',1),('nondep_feat.pkl',0)]:
    with open(f'{R}/{fn}','rb') as f: d=pickle.load(f)
    for name,vids in zip(d['name'],d['features']):
        sp=split_of.get(name)
        if sp is None: continue
        out=[]
        for v in vids[:360]:
            v=np.asarray(v, dtype=np.float32)[:60,:]
            if NORM=='norm':
                w=np.zeros_like(v)
                w[:,:136]=nf(v[:,:136].astype(np.float64)).astype(np.float32)
                w[:,136:]=v[:,136:]
                v=w
            out.append(v)
        data[sp].append((out,y,name))
    del d
print(f'mode={NORM}', {k:(len(v),sum(y for _,y,_ in v)) for k,v in data.items()}, flush=True)

def collate(batch):
    feats,lengths=[],[]
    for vids,_,_ in batch:
        ts=[]
        for f in vids:
            f=torch.from_numpy(np.asarray(f,dtype=np.float32))
            if f.shape[0]<60: f=torch.cat([f,torch.zeros(60-f.shape[0],f.shape[1])],0)
            if f.shape[1]<161: f=torch.cat([f,torch.zeros(f.shape[0],161-f.shape[1])],1)
            ts.append(f[:60,:161])
        lengths.append(len(ts)); feats.append(pad_sequence(ts,batch_first=True))
    x=pad_sequence(feats,batch_first=True)
    return (x,lengths,[b[2] for b in batch]), torch.tensor([b[1] for b in batch],dtype=torch.float32)

def batches(split,bs,shuffle):
    idx=list(range(len(data[split])))
    if shuffle: random.shuffle(idx)
    for i in range(0,len(idx),bs): yield collate([data[split][j] for j in idx[i:i+bs]])

def evaluate(net,split,lf):
    net.eval(); P,Y,L,n=[],[],0.0,0
    with torch.no_grad():
        for (x,ln,nm),y in batches(split,BS,False):
            lg=net((x.to(DEV),ln,nm)).view(-1,1); yy=y.to(DEV).view(-1,1)
            L+=lf(lg,yy).item()*yy.numel(); n+=yy.numel()
            P+=(torch.sigmoid(lg)>0.5).view(-1).long().cpu().tolist(); Y+=yy.view(-1).long().cpu().tolist()
    return dict(loss=L/n,acc=accuracy_score(Y,P),f1=f1_score(Y,P,zero_division=0))

res=[]
for seed in [110,42,7,0,1]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net=LSTM_han(d=256,t_downsample=4).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=LR); lf=nn.BCEWithLogitsLoss()
    best=(1e9,None)
    for ep in range(EPOCHS):
        net.train()
        for (x,ln,nm),y in batches('train',BS,True):
            opt.zero_grad(); lf(net((x.to(DEV),ln,nm)).view(-1,1), y.to(DEV).view(-1,1)).backward(); opt.step()
        v=evaluate(net,'val',lf)
        if v['loss']<=best[0]: best=(v['loss'],{k:t.detach().clone() for k,t in net.state_dict().items()})
    net.load_state_dict(best[1]); t=evaluate(net,'test',lf)
    print(f">> [{NORM}] seed{seed} TEST f1={t['f1']:.4f} acc={t['acc']:.4f}", flush=True)
    res.append((t['f1'],t['acc']))
f=np.array([r[0] for r in res]); a=np.array([r[1] for r in res])
print(f"\n===== baseline LSTM_han, 입력={NORM} ({len(res)}seed) =====", flush=True)
print(f"  test F1 = {f.mean():.4f} ± {f.std():.4f}   acc = {a.mean():.4f} ± {a.std():.4f}", flush=True)
print("  [참고] raw입력 3seed=0.6885 / 얼굴위치·크기만 로지스틱=0.6875 / 우리 협응=0.7468", flush=True)
print('DONE', flush=True)
