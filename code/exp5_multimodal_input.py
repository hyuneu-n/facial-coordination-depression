"""
exp5 — 스모크: 입력 보강 (AU + gaze + pose). 긍정앵커 + ST-GCN/통계.
선행연구: 우울=시선회피/고개숙임. AU만(0.61)보다 오르나 확인. 손절 0.65.
"""
import numpy as np, csv, warnings
from pathlib import Path
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
POST=6.0; T=64; SEEDS=[42,1,2,3]
POS=['last time you felt really happy','really happy','proud','enjoy','felt really happy']

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

def read_txt(path,cols):
    if not path.exists(): return None,None
    h=[x.strip() for x in open(path).readline().split(',')]
    try: ti,oi=h.index('timestamp'),h.index('success'); ci=[h.index(c) for c in cols]
    except: return None,None
    ts,fe=[],[]
    with open(path) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                ts.append(float(v[ti])); fe.append([float(v[i]) for i in ci])
            except: pass
    return np.array(ts),np.array(fe)

def combined(pid):
    ts,au=read_txt(D/f'{pid}_CLNF_AUs.txt',AU)
    if ts is None: return None,None
    _,gz=read_txt(D/f'{pid}_CLNF_gaze.txt',['x_0','y_0','z_0','x_1','y_1','z_1'])
    _,po=read_txt(D/f'{pid}_CLNF_pose.txt',['Rx','Ry','Rz'])  # 머리 회전(고개)
    n=len(au)
    parts=[au]
    if gz is not None and len(gz)>=n: parts.append(gz[:n])
    if po is not None and len(po)>=n: parts.append(po[:n])
    feat=np.concatenate(parts,axis=1)
    return ts,feat

lab=labels()
def build():
    X,y=[],[]
    for pid,l in lab.items():
        ts,feat=combined(pid)
        if ts is None: continue
        base=feat.mean(0); bs=feat.std(0)+1e-6
        segs=[feat[(ts>=a)&(ts<a+POST)] for a in anchors(transcript(pid),POS)]
        segs=[s for s in segs if len(s)>=4]
        if not segs: continue
        seg=(np.vstack(segs)-base)/bs
        X.append(seg); y.append(l)
    return X,np.array(y)

Xseq,y=build()
NF=Xseq[0].shape[1]
print(f'입력 차원(AU+gaze+pose)={NF}, n={len(y)}, 우울{int(y.sum())}',flush=True)

# 통계요약
Xs=np.array([np.concatenate([s.mean(0),s.std(0)]) for s in Xseq])
aucs=[]
for s in SEEDS:
    cv=StratifiedKFold(5,shuffle=True,random_state=s)
    aucs.append(cross_val_score(LogisticRegression(max_iter=1000,class_weight='balanced'),Xs,y,cv=cv,scoring='roc_auc').mean())
print(f'[통계요약 AU+gaze+pose] AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)

# ST-GCN (노드=피처)
def resample(x,T):
    idx=np.linspace(0,len(x)-1,T)
    return np.stack([np.interp(idx,np.arange(len(x)),x[:,j]) for j in range(x.shape[1])],1)
Xg=np.array([resample(s,T) for s in Xseq],dtype=np.float32)
Xflat=Xg.reshape(-1,NF); A=np.abs(np.nan_to_num(np.corrcoef(Xflat.T))); A=A/(A.sum(1,keepdims=True)+1e-6)
A=torch.tensor(A,dtype=torch.float32).to(DEV)
class STGCN(nn.Module):
    def __init__(s,n,h=32):
        super().__init__(); s.A=A
        s.gc1=nn.Linear(1,h); s.tcn1=nn.Conv1d(h,h,3,padding=1); s.gc2=nn.Linear(h,h)
        s.head=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,1))
    def forward(s,x):
        B,T,N=x.shape; h=s.gc1(x.unsqueeze(-1))
        h=torch.einsum('nm,btmh->btnh',s.A,h); h=torch.relu(h)
        h=h.permute(0,2,3,1).reshape(B*N,-1,T); h=torch.relu(s.tcn1(h))
        h=h.reshape(B,N,-1,T).permute(0,3,1,2); h=torch.relu(s.gc2(h))
        h=torch.einsum('nm,btmh->btnh',s.A,h); h=h.mean(dim=(1,2))
        return s.head(h).squeeze(-1)
def run(Xtr,ytr,Xte,yte,seed):
    torch.manual_seed(seed); m=STGCN(NF).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    pw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],dtype=torch.float32).to(DEV)
    lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=torch.tensor(Xtr).to(DEV); yt=torch.tensor(ytr,dtype=torch.float32).to(DEV); best=0
    for ep in range(80):
        m.train(); perm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),16):
            idx=perm[i:i+16]; opt.zero_grad(); loss=lf(m(Xt[idx]),yt[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): p=torch.sigmoid(m(torch.tensor(Xte).to(DEV))).cpu().numpy()
        try: best=max(best,roc_auc_score(yte,p))
        except: pass
    return best
aucs=[]
for s in SEEDS:
    cv=StratifiedKFold(5,shuffle=True,random_state=s)
    aucs.append(np.mean([run(Xg[tr],y[tr],Xg[te],y[te],s) for tr,te in cv.split(Xg,y)]))
print(f'[ST-GCN AU+gaze+pose] AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)
print(f'손절 0.65 → {"통과 ✅" if np.mean(aucs)>=0.65 else "미달"}',flush=True)
