"""exp105 — 조기탐지 곡선 다코퍼스 (D-Vlog·LMVD·E-DAIC). visual 협응(SPD+GRU) 통일.
앞 20/40/60/80/100%만 보고 AUC. '앞부분이면 충분'이 보편인가? fig8.
"""
import numpy as np, warnings, csv, os, re, glob
import pandas as pd
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
dev='cuda'; T=256; W=64; STR=32; EP=55; NSEED=4; FRACS=[0.2,0.4,0.6,0.8,1.0]
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
def resamp(X,t=T):
    n=len(X);idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def win_covs(seq,d):
    cs=[]
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
        Bs,K,dd,_=covs.shape;v=self.sl(covs.reshape(Bs*K,dd,dd)).reshape(Bs,K,-1);return self.pre(v)
class Model(nn.Module):
    def __init__(self,d):
        super().__init__();self.spd=SPDseq(d);self.gru=nn.GRU(64,32,batch_first=True,bidirectional=True)
        self.cls=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs):o,_=self.gru(self.spd(covs));return self.cls(o.mean(1)).squeeze(-1)
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
def early_curve(RAW,Y,split,d,name):
    # RAW: list of (T_i, d) 연속 시계열. split: 'train'/'valid'/'test'. 앞frac% → cov
    Y=np.array(Y);split=np.array(split)
    def build(frac):
        out=[]
        for x in RAW:
            n=max(W+2,int(len(x)*frac));xx=x[:n]
            xx=(xx-xx.mean(0))/(xx.std(0)+1e-6);out.append(win_covs(resamp(xx),d))
        return np.array(out)
    FULL=build(1.0);PREF={f:build(f) for f in FRACS if f<1.0};PREF[1.0]=FULL
    tr=split=='train';va=split=='valid';te=split=='test'
    res={f:[] for f in FRACS}
    for seed in range(NSEED):
        torch.manual_seed(seed);net=Model(d).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
        pw=T3([(Y[tr]==0).sum()/max(1,(Y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Ct=T3(FULL[tr]);yt=T3(Y[tr]);Cv=T3(FULL[va])
        n=tr.sum();bs=64;bva=0;bs_state=None
        for ep in range(EP):
            net.train();perm=torch.randperm(n)
            for i in range(0,n,bs):
                b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Ct[b]),yt[b]);loss.backward();opt.step()
            net.eval()
            with torch.no_grad():a=roc_auc_score(Y[va],torch.sigmoid(net(Cv)).cpu().numpy())
            if a>bva:bva=a;bs_state={k:v.detach().clone() for k,v in net.state_dict().items()}
        net.load_state_dict(bs_state);net.eval()
        for f in FRACS:
            Ce=T3(PREF[f][te])
            with torch.no_grad():pt=torch.sigmoid(net(Ce)).cpu().numpy()
            res[f].append(roc_auc_score(Y[te],pt))
    ms=[np.mean(res[f]) for f in FRACS]
    print(f'[{name}] '+' '.join(f'{int(f*100)}%={m:.3f}' for f,m in zip(FRACS,ms)),flush=True)
    return ms
curves={}
# --- D-Vlog (landmark PCA20) ---
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
RAW=[];Y=[];SP=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fold=r.get('fold','').strip().lower()
    if 'val' in fold:fold='valid'
    if fold not in('train','valid','test'):continue
    fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:V=np.load(fv)
    except:continue
    if V.ndim!=2 or V.shape[0]<W+8 or V.shape[1]!=136:continue
    RAW.append(nf(V));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0);SP.append(fold)
pca=PCA(20).fit(np.vstack([RAW[i][::4] for i in range(len(RAW)) if SP[i]=='train']))
RAWp=[pca.transform(x) for x in RAW]
curves['D-Vlog']=early_curve(RAWp,Y,SP,20,'D-Vlog')
# --- E-DAIC (per-second AU) ---
lab={}
for spn in ['train','dev','test']:
    p=f'/home/hyuneun/disk_b/🟡facial-prodrome/data/E-DAIC/labels/{spn}_split.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            pid=r.get('Participant_ID','').strip();b=r.get('PHQ_Binary','').strip()
            if pid and b in('0','1'):lab[pid]=(int(b),'valid' if spn=='dev' else spn)
def persec(f,usec):
    try:df=pd.read_csv(f,usecols=usec)
    except:return None
    df.columns=[c.strip() for c in df.columns]
    if 'timestamp' not in df or not set(AUS).issubset(df.columns):return None
    if 'success' in df:df=df[df['success']==1]
    if len(df)<80:return None
    df=df.copy();df['sec']=np.floor(df['timestamp']).astype(int)
    return df.groupby('sec')[AUS].mean().values
RAW=[];Y=[];SP=[];usc=lambda c:c.strip() in set(['timestamp','success','confidence']+AUS)
for dfold in sorted(glob.glob('/home/hyuneun/disk_b/🟡facial-prodrome/data/E-DAIC/extracted/*_P/')):
    pid=os.path.basename(dfold.rstrip('/')).replace('_P','')
    if pid not in lab:continue
    f=dfold+f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'
    if not os.path.exists(f):continue
    ps=persec(f,usc)
    if ps is None or len(ps)<W+8:continue
    RAW.append(ps);Y.append(lab[pid][0]);SP.append(lab[pid][1])
if len(set(SP))>=3 and (np.array(SP)=='test').sum()>5:
    curves['E-DAIC']=early_curve(RAW,Y,SP,17,'E-DAIC')
# --- LMVD (per-second AU, 자체 split) ---
def lm(i):return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
RAW=[];Y=[];SP=[]
for f in sorted(glob.glob('/home/hyuneun/disk_b/🟡facial-prodrome/data/LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(f))
    if not m:continue
    i=int(m.group(1));ps=persec(f,usc)
    if ps is None or len(ps)<W+8:continue
    RAW.append(ps);Y.append(lm(i));SP.append('train' if i%5<3 else('valid' if i%5==3 else 'test'))
if len(set(SP))>=3:
    curves['LMVD']=early_curve(RAW,Y,SP,17,'LMVD')
# ==== fig8 ====
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for c in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c):fm.fontManager.addfont(c);plt.rcParams['font.family']=fm.FontProperties(fname=c).get_name()
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(6.6,4.6));x=[f*100 for f in FRACS]
cols={'D-Vlog':'#1565C0','LMVD':'#2E7D32','E-DAIC':'#d97706'}
for nm,ms in curves.items():
    plt.plot(x,ms,marker='o',lw=2.2,label=nm,color=cols.get(nm,'#555'))
    plt.text(x[-1]+1,ms[-1],nm,color=cols.get(nm,'#555'),fontsize=10,va='center',fontweight='bold')
plt.axhline(0.5,ls=':',color='#c0392b',lw=1);plt.text(21,0.505,'chance',color='#c0392b',fontsize=9)
plt.xlabel('관찰한 세션 앞부분 (%)');plt.ylabel('우울 탐지 AUC (visual 협응)')
plt.title('조기 탐지 곡선 — 다코퍼스\n세션 앞부분만으로 우울이 잡히나',fontsize=12)
plt.ylim(0.5,0.9);plt.legend(fontsize=9,loc='lower right');plt.tight_layout()
plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig8_early_multi.png',dpi=145,bbox_inches='tight')
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp105_early_multi.csv','w') as fo:
    fo.write('corpus,'+','.join(f'f{int(f*100)}' for f in FRACS)+'\n')
    for nm,ms in curves.items():fo.write(nm+','+','.join(f'{m:.4f}' for m in ms)+'\n')
print('DONE fig8_early_multi.png',flush=True)
