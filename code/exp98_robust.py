"""exp98 — 시간모델(GRU) 방법 굳히기. (1)D-Vlog 다seed robustness: mean vs GRU 페어드.
(2)E-DAIC 일반화(visual-only): 시간집계가 다른 벤치마크서도 도움? 마커재현 아니라 방법일반화.
"""
import numpy as np, warnings, csv, os, re, glob
import pandas as pd
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings('ignore')
dev='cuda'; T=256; W=64; STR=32; EP=60; NSEED=10
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
def resamp(X,t=T):
    n=len(X);idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def win_covs(seq):
    d=seq.shape[1];cs=[]
    for s in range(0,T-W+1,STR):
        seg=seq[s:s+W];seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);cs.append(c+1e-3*np.eye(d))
    return np.array(cs)
class SPDseq(nn.Module):
    def __init__(self,d,out=16):
        super().__init__();self.W=nn.Parameter(torch.randn(out,d)*0.1);self.out=out
        self.pre=nn.Sequential(nn.Linear(out*(out+1)//2,64),nn.ReLU())
    def sl(self,M):
        I=torch.eye(self.out,device=M.device,dtype=torch.float64)
        Bm=(self.W.double()@M.double()@self.W.double().transpose(0,1))+1e-2*I
        ev,U=torch.linalg.eigh(Bm);ev=ev.clamp(min=1e-3);M2=U@torch.diag_embed(ev)@U.transpose(1,2)
        ev2,U2=torch.linalg.eigh(M2+1e-4*I);L=U2@torch.diag_embed(torch.log(ev2.clamp(min=1e-4)))@U2.transpose(1,2)
        idx=torch.triu_indices(self.out,self.out);return L[:,idx[0],idx[1]].float()
    def forward(self,covs):
        Bs,K,d,_=covs.shape;v=self.sl(covs.reshape(Bs*K,d,d)).reshape(Bs,K,-1);return self.pre(v)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Model(nn.Module):
    def __init__(self,d,Ca,kind,use_audio):
        super().__init__();self.spd=SPDseq(d);self.kind=kind;self.use_audio=use_audio
        if kind=='gru':self.gru=nn.GRU(64,32,batch_first=True,bidirectional=True)
        aud=96 if use_audio else 0
        if use_audio:self.a=CNNb(Ca)
        self.cls=nn.Sequential(nn.Linear(64+aud,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def agg(self,x):
        if self.kind=='gru':o,_=self.gru(x);return o.mean(1)
        return x.mean(1)
    def forward(self,covs,xa=None):
        h=self.agg(self.spd(covs))
        if self.use_audio:h=torch.cat([h,self.a(xa)],1)
        return self.cls(h).squeeze(-1)
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
def train_eval(COVS,AUD,Y,tr,va,te,d,Ca,kind,use_audio,seed):
    torch.manual_seed(seed);net=Model(d,Ca,kind,use_audio).to(dev)
    opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Ct=T3(COVS[tr]);yt=T3(Y[tr]);Cv=T3(COVS[va]);Ce=T3(COVS[te])
    At=T3(AUD[tr]).transpose(1,2) if use_audio else None
    Av=T3(AUD[va]).transpose(1,2) if use_audio else None
    Ae=T3(AUD[te]).transpose(1,2) if use_audio else None
    n=tr.sum();bs=64;bva=0;bp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad()
            loss=lf(net(Ct[b],At[b] if use_audio else None),yt[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            pv=torch.sigmoid(net(Cv,Av)).cpu().numpy();pt=torch.sigmoid(net(Ce,Ae)).cpu().numpy()
        try:
            a=roc_auc_score(Y[va],pv)
            if a>bva:bva=a;bp=pt
        except:pass
    return roc_auc_score(Y[te],bp),f1_score(Y[te],(bp>0.5).astype(int))
# ========== (1) D-Vlog robustness ==========
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
raw=[];araw=[];Y=[];Fd=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fold=r.get('fold','').strip().lower()
    if 'val' in fold:fold='valid'
    if fold not in('train','valid','test'):continue
    fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
    if not(fv.exists() and fa.exists()):continue
    try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
    except:continue
    if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136 or Au.ndim!=2 or len(Au)<10:continue
    raw.append(nf(V));araw.append(Au);Y.append(1 if r['label'].strip().lower().startswith('depress') else 0);Fd.append(fold)
Y=np.array(Y);Fd=np.array(Fd)
pca=PCA(20).fit(np.vstack([raw[i][::3] for i in range(len(raw)) if Fd[i]=='train']))
COVS=[];AUD=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);COVS.append(win_covs(resamp(Zz)))
    Aa=(araw[i]-araw[i].mean(0))/(araw[i].std(0)+1e-6);AUD.append(resamp(Aa))
COVS=np.array(COVS);AUD=np.array(AUD);Ca=AUD.shape[2]
tr=Fd=='train';va=Fd=='valid';te=Fd=='test'
print(f'=== (1) D-Vlog robustness: seed별 mean vs GRU (n_test={te.sum()}) ===',flush=True)
mm=[];gg=[];win=0
for s in range(NSEED):
    am,fm=train_eval(COVS,AUD,Y,tr,va,te,20,Ca,'mean',True,s)
    ag,fg=train_eval(COVS,AUD,Y,tr,va,te,20,Ca,'gru',True,s)
    mm.append(am);gg.append(ag);win+=int(ag>am)
    print(f'  seed{s}: mean={am:.4f} gru={ag:.4f} {"GRU승" if ag>am else "mean승"}',flush=True)
print(f'  >> mean {np.mean(mm):.4f}±{np.std(mm):.4f} | GRU {np.mean(gg):.4f}±{np.std(gg):.4f} | GRU 승률 {win}/{NSEED}',flush=True)
# ========== (2) E-DAIC 일반화 (visual-only) ==========
lab={}
for sp in ['train','dev','test']:
    p=f'/home/hyuneun/disk_b/🟡facial-prodrome/data/E-DAIC/labels/{sp}_split.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            pid=r.get('Participant_ID','').strip();b=r.get('PHQ_Binary','').strip()
            if pid and b in('0','1'):lab[(pid)]=(int(b),sp if sp!='dev' else 'valid')
COVe=[];Ye=[];Fe=[]
for dfold in sorted(glob.glob('/home/hyuneun/disk_b/🟡facial-prodrome/data/E-DAIC/extracted/*_P/')):
    pid=os.path.basename(dfold.rstrip('/')).replace('_P','')
    if pid not in lab:continue
    f=dfold+f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'
    if not os.path.exists(f):continue
    try:df=pd.read_csv(f,usecols=lambda c:c.strip() in set(['timestamp','success','confidence']+AUS))
    except:continue
    df.columns=[c.strip() for c in df.columns]
    if 'success' in df:df=df[df['success']==1]
    if len(df)<80 or not set(AUS).issubset(df.columns):continue
    X=df[AUS].values.astype(float);X=(X-X.mean(0))/(X.std(0)+1e-6)
    COVe.append(win_covs(resamp(X)));Ye.append(lab[pid][0]);Fe.append(lab[pid][1])
COVe=np.array(COVe);Ye=np.array(Ye);Fe=np.array(Fe)
tre=Fe=='train';vae=Fe=='valid';tee=Fe=='test'
print(f'\n=== (2) E-DAIC 일반화 visual-only (n={len(Ye)}, test={tee.sum()}) ===',flush=True)
if tee.sum()>5 and vae.sum()>2:
    mm2=[];gg2=[];win2=0
    for s in range(5):
        am,_=train_eval(COVe,None,Ye,tre,vae,tee,17,0,'mean',False,s)
        ag,_=train_eval(COVe,None,Ye,tre,vae,tee,17,0,'gru',False,s)
        mm2.append(am);gg2.append(ag);win2+=int(ag>am)
    print(f'  mean {np.mean(mm2):.4f}±{np.std(mm2):.4f} | GRU {np.mean(gg2):.4f}±{np.std(gg2):.4f} | GRU 승률 {win2}/5',flush=True)
else:
    print('  E-DAIC split 부족',flush=True)
print('DONE',flush=True)
