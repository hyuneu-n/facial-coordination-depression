"""
exp86 — SPD-manifold fusion 모델 다코퍼스 검증. (DISFA는 우울라벨 없어 제외)
CMDC(visual, CV) / LMVD(+VGGish, CV) / E-DAIC(+eGeMAPS, 공식) / DAIC(+COVAREP, 공식).
모델: SPDNet(visual, window 공분산) + audio CNN fusion(있으면). 결과: results/exp86_cross_spd.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); dev='cuda'; T=256; W=64; STR=32; EP=55
def resamp(X,t=T):
    n=len(X)
    if n<3:return np.zeros((t,X.shape[1]))
    idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
def win_covs(seq):
    d=seq.shape[1];cs=[]
    for s in range(0,T-W+1,STR):
        seg=seq[s:s+W];seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);cs.append(c+1e-3*np.eye(d))
    return np.array(cs)
class SPDNet(nn.Module):
    def __init__(self,d,out=16):
        super().__init__();self.W=nn.Parameter(torch.randn(out,d)*0.1);self.out=out
        self.pre=nn.Sequential(nn.Linear(out*(out+1)//2,64),nn.ReLU())
    def sl(self,M):
        Bm=self.W@M@self.W.transpose(0,1)+1e-4*torch.eye(self.out,device=M.device)
        ev,U=torch.linalg.eigh(Bm);ev=ev.clamp(min=1e-4);M2=U@torch.diag_embed(ev)@U.transpose(1,2)
        ev2,U2=torch.linalg.eigh(M2+1e-6*torch.eye(self.out,device=M.device))
        L=U2@torch.diag_embed(torch.log(ev2.clamp(min=1e-6)))@U2.transpose(1,2)
        idx=torch.triu_indices(self.out,self.out);return L[:,idx[0],idx[1]]
    def forward(self,covs):
        Bs,K,d,_=covs.shape;v=self.sl(covs.reshape(Bs*K,d,d)).reshape(Bs,K,-1);return self.pre(v).mean(1)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Model(nn.Module):
    def __init__(self,d,Ca):
        super().__init__();self.spd=SPDNet(d);fin=64
        self.has_a=Ca>0
        if self.has_a:self.a=CNNb(Ca);fin+=96
        self.head=nn.Sequential(nn.Linear(fin,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,xa):
        h=self.spd(covs)
        if self.has_a:h=torch.cat([h,self.a(xa)],1)
        return self.head(h).squeeze(-1)
def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
def fit_eval(Ctr,Atr,ytr,Cte,Ate,yte,d,Ca,seed):
    torch.manual_seed(seed);net=Model(d,Ca).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
    pw=T3([(ytr==0).sum()/max(1,(ytr==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    C=T3(Ctr);A=T3(Atr).transpose(1,2) if Ca else None;y=T3(ytr)
    Ce=T3(Cte);Ae=T3(Ate).transpose(1,2) if Ca else None;n=len(ytr);bs=64;best=0.5;bp=None
    for ep in range(EP):
        net.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs];opt.zero_grad();loss=lf(net(C[b],A[b] if Ca else None),y[b]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():p=torch.sigmoid(net(Ce,Ae if Ca else None)).cpu().numpy()
        try:
            a=roc_auc_score(yte,p)
            if a>best:best=a;bp=p
        except:pass
    return best,bp
def eval_cv(COVS,AUD,y,d,Ca,name):
    y=np.array(y);aucs=[];f1s=[]
    for seed in range(3):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(COVS,y):
            a,bp=fit_eval(COVS[tr],AUD[tr] if Ca else np.zeros((tr.sum(),1,1)),y[tr],COVS[te],AUD[te] if Ca else np.zeros((te.sum(),1,1)),y[te],d,Ca,seed)
            aucs.append(a)
            if bp is not None:f1s.append(f1_score(y[te],(bp>0.5).astype(int)))
    print(f'  {name}: AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} F1={np.mean(f1s):.3f}',flush=True)
    return np.mean(aucs),np.mean(f1s)
def eval_split(COVS,AUD,y,fold,d,Ca,name,evfold):
    tr=fold=='train';ev=fold==evfold;y=np.array(y);aucs=[];f1s=[]
    for seed in range(5):
        a,bp=fit_eval(COVS[tr],(AUD[tr] if Ca else np.zeros((tr.sum(),1,1))),y[tr],COVS[ev],(AUD[ev] if Ca else np.zeros((ev.sum(),1,1))),y[ev],d,Ca,seed)
        aucs.append(a)
        if bp is not None:f1s.append(f1_score(y[ev],(bp>0.5).astype(int)))
    print(f'  {name}: AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} F1={np.mean(f1s):.3f}',flush=True)
    return np.mean(aucs),np.mean(f1s)
out=[]
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
# CMDC (visual, CV)
print('CMDC...',flush=True)
C=B/'CMDC/extracted';wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID=hd.index('ID');iMDD=hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
CO=[];y=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=W:
        Z=np.vstack(parts);Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);CO.append(win_covs(resamp(Zz)));y.append(l)
CO=np.array(CO)
a,f=eval_cv(CO,None,y,len(AUc),0,'CMDC(visual)');out.append(['CMDC',len(y),a,f])
# LMVD (+VGGish, CV)
print('LMVD...',flush=True)
V=B/'LMVD/extracted/Video_feature';Au=B/'LMVD/extracted/Audio_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423):return 1
    if (602<=i<=1116) or (1425<=i<=1824):return 0
    return None
CO=[];AD=[];y=[]
for f in sorted(glob.glob(str(V/'*.csv'))):
    sid=Path(f).stem;l=lab(int(sid))
    if l is None:continue
    ap=Au/f'{sid}.npy'
    if not ap.exists():continue
    h=[x.strip() for x in open(f).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:continue
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    if len(fe)<W:continue
    try:au=np.nan_to_num(np.load(ap).astype(float))
    except:continue
    if au.ndim!=2 or len(au)<10:continue
    Z=np.nan_to_num(np.array(fe));Zz=(Z-Z.mean(0))/(Z.std(0)+1e-6);CO.append(win_covs(resamp(Zz)))
    AD.append(resamp((au-au.mean(0))/(au.std(0)+1e-6)));y.append(l)
CO=np.array(CO);AD=np.array(AD);Ca=AD.shape[2]
a,f=eval_cv(CO,AD,y,len(AUc),Ca,'LMVD(+audio)');out.append(['LMVD',len(y),a,f])
# E-DAIC (+eGeMAPS, 공식 test)
print('E-DAIC...',flush=True)
E=B/'E-DAIC';esp={}
for fn,fo in [('train_split.csv','train'),('dev_split.csv','dev'),('test_split.csv','test')]:
    p=E/'labels'/fn
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):esp[pid]=(int(float(b)),fo)
def eload(pid):
    ff=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*.csv'))
    af=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenSMILE*egemaps*.csv'))
    if not(ff and af):return None
    h=[x.strip() for x in open(ff[0]).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    au=[]
    for ln in open(ff[0]).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            au.append([float(v[i]) for i in ai])
        except:pass
    if len(au)<W:return None
    au=np.array(au);Zz=(au-au.mean(0))/(au.std(0)+1e-6)
    eg=[]
    for ln in open(af[0]).readlines()[1:]:
        parts=ln.split(';') if ';' in ln else ln.split(',')
        try:eg.append([float(x) for x in parts[1:] if x.strip()!=''])
        except:pass
    if len(eg)<10:return None
    L=len(eg[0]);eg=np.array([r for r in eg if len(r)==L]);Az=(eg-eg.mean(0))/(eg.std(0)+1e-6)
    return win_covs(resamp(Zz)),resamp(Az)
CO=[];AD=[];y=[];fold=[]
for pid,(l,fo) in esp.items():
    d=eload(pid)
    if d is None:continue
    CO.append(d[0]);AD.append(d[1]);y.append(l);fold.append(fo)
CO=np.array(CO);AD=np.array(AD);fold=np.array(fold);Ca=AD.shape[2]
a,f=eval_split(CO,AD,y,fold,len(AUc),Ca,'E-DAIC(+audio,test)','test');out.append(['E-DAIC',int((fold=="test").sum()),a,f])
print('\n[비교] D-Vlog SPD+audio 0.808/0.788 | 원논문류 임상 얼굴 baseline~0.6',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp86_cross_spd.csv','w') as fo_:
    fo_.write('corpus,n,AUC,F1\n')
    for r in out:fo_.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp86_cross_spd.csv',flush=True)
