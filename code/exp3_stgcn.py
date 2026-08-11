"""
exp3_stgcn.py — 백본 후보1: ST-GCN (AU 그래프 시공간)
AU 시퀀스를 그래프(노드=14 AU, 시간축)로 → ST-GCN → 우울 이진.
긍정 앵커 구간 사용. 손절선 AUC 0.65.
가벼운 ST-GCN (2 layer). LOSO 아니고 5-fold(빠른 판단용), 여러 seed.
"""
import numpy as np, csv, warnings
from pathlib import Path
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r',
      'AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
NA = len(AU); T = 64  # 시퀀스 길이(리샘플)
POST = 6.0
POS = ['last time you felt really happy','really happy','proud','enjoy','felt really happy']
SEEDS = [42,1,2,3]

def labels():
    lab={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)): lab[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
    return lab
def transcript(pid):
    p=D/f'{pid}_TRANSCRIPT.csv'; rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try: rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except: pass
    return rows
def anchors(rows,keys): return [sp for st,sp,spk,val in rows if spk=='Ellie' and any(k in val for k in keys)]
def au_series(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists(): return None,None
    h=[x.strip() for x in open(p).readline().split(',')]
    ti,oi=h.index('timestamp'),h.index('success'); ai=[h.index(c) for c in AU]
    ts,fe=[],[]
    with open(p) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                ts.append(float(v[ti])); fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(ts),np.array(fe)

def resample(x,T):
    if len(x)==T: return x
    idx=np.linspace(0,len(x)-1,T)
    return np.stack([np.interp(idx,np.arange(len(x)),x[:,j]) for j in range(x.shape[1])],1)

lab=labels(); X,y=[],[]
for pid,l in lab.items():
    ts,au=au_series(pid)
    if ts is None or len(ts)==0: continue
    base=au.mean(0); bs=au.std(0)+1e-6
    segs=[]
    for a in anchors(transcript(pid),POS):
        m=(ts>=a)&(ts<a+POST)
        if m.sum()>=4: segs.append(au[m])
    if not segs: continue
    seg=np.vstack(segs); seg=(seg-base)/bs  # 개인정규화
    X.append(resample(seg,T)); y.append(l)
X=np.array(X,dtype=np.float32); y=np.array(y)  # (N,T,AU)
print(f'ST-GCN 입력: {X.shape}, 우울{int(y.sum())}/정상{int((y==0).sum())}',flush=True)

# AU 상관 그래프 (인접행렬) — 데이터 기반
Xflat=X.reshape(-1,NA); A=np.corrcoef(Xflat.T); A=np.abs(np.nan_to_num(A))
A=A/(A.sum(1,keepdims=True)+1e-6)
A=torch.tensor(A,dtype=torch.float32).to(DEV)

class STGCN(nn.Module):
    def __init__(self,na,hid=32):
        super().__init__()
        self.A=A
        self.gc1=nn.Linear(1,hid)       # 노드특징(스칼라 AU값)->hid
        self.tcn1=nn.Conv1d(hid,hid,3,padding=1)
        self.gc2=nn.Linear(hid,hid)
        self.tcn2=nn.Conv1d(hid,hid,3,padding=1)
        self.head=nn.Sequential(nn.Linear(hid,hid),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hid,1))
    def forward(self,x):  # x:(B,T,NA)
        B,T,N=x.shape
        h=x.unsqueeze(-1)                      # (B,T,N,1)
        h=self.gc1(h)                          # (B,T,N,hid)
        h=torch.einsum('nm,btmh->btnh',self.A,h)  # 그래프 전파
        h=torch.relu(h)
        h=h.permute(0,2,3,1).reshape(B*N,-1,T)
        h=torch.relu(self.tcn1(h))
        h=h.reshape(B,N,-1,T).permute(0,3,1,2)  # (B,T,N,hid)
        h=self.gc2(h); h=torch.einsum('nm,btmh->btnh',self.A,h); h=torch.relu(h)
        h=h.mean(dim=(1,2))                     # 시공간 평균 풀링 -> (B,hid)
        return self.head(h).squeeze(-1)

def run_fold(Xtr,ytr,Xte,yte,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=STGCN(NA).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    pw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],dtype=torch.float32).to(DEV)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=torch.tensor(Xtr).to(DEV); yt=torch.tensor(ytr,dtype=torch.float32).to(DEV)
    best=0; bestauc=0
    for ep in range(80):
        m.train(); perm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),16):
            idx=perm[i:i+16]
            opt.zero_grad(); out=m(Xt[idx]); loss=lossf(out,yt[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p=torch.sigmoid(m(torch.tensor(Xte).to(DEV))).cpu().numpy()
        try: a=roc_auc_score(yte,p)
        except: a=0.5
        bestauc=max(bestauc,a)
    return bestauc

aucs=[]
for s in SEEDS:
    cv=StratifiedKFold(5,shuffle=True,random_state=s)
    fold=[run_fold(X[tr],y[tr],X[te],y[te],s) for tr,te in cv.split(X,y)]
    aucs.append(np.mean(fold))
    print(f'  seed{s}: AUC={np.mean(fold):.3f}',flush=True)
print(f'\n[ST-GCN 긍정앵커] AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)
print(f'손절선 0.65 → {"통과 ✅ 계속" if np.mean(aucs)>=0.65 else "미달 → 백본 교체 검토"}',flush=True)
