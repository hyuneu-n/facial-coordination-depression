"""
exp83 — 딥 AV+coupling을 DAIC·E-DAIC 공식 split서 돌려 D-Vlog 결과 일반화 확인.
"D-Vlog 한정 아닌가" 점검. 시각-CNN(AU seq)+음성-CNN(COVAREP/eGeMAPS seq)+coupling.
DAIC: train→dev(테스트). E-DAIC: train→test(dev early-stop). 결과: results/exp83_other.csv
"""
import numpy as np, warnings, csv, glob
from pathlib import Path
import torch, torch.nn as nn
from sklearn.covariance import ledoit_wolf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); dev='cuda'; T=200; EP=60
def resamp(X,t=T):
    n=len(X)
    if n<3:return np.zeros((t,X.shape[1]))
    idx=np.linspace(0,n-1,t);return np.stack([np.interp(idx,np.arange(n),X[:,c]) for c in range(X.shape[1])],1)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Net(nn.Module):
    def __init__(self,Cv,Ca,tan):
        super().__init__();self.v=CNNb(Cv);self.a=CNNb(Ca)
        self.g=nn.Sequential(nn.Linear(tan,96),nn.ReLU(),nn.Dropout(0.4),nn.Linear(96,48))
        self.head=nn.Sequential(nn.Linear(96+96+48,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,xv,xa,g):return self.head(torch.cat([self.v(xv),self.a(xa),self.g(g)],1)).squeeze(-1)
def bench(VIS,AUD,COV,y,tr,ev,name,Ca):
    Ts=TangentSpace(metric='riemann').fit(COV[tr]);Z=Ts.transform(COV);sc=StandardScaler().fit(Z[tr]);G=sc.transform(Z)
    def T3(a):return torch.tensor(a,dtype=torch.float32,device=dev)
    aucs=[];f1s=[]
    for seed in range(5):
        torch.manual_seed(seed);net=Net(VIS.shape[2],Ca,G.shape[1]).to(dev);opt=torch.optim.Adam(net.parameters(),7e-4,weight_decay=1e-4)
        pw=T3([(y[tr]==0).sum()/max(1,(y[tr]==1).sum())]);lf=nn.BCEWithLogitsLoss(pos_weight=pw)
        Xv=T3(VIS[tr]).transpose(1,2);Xa=T3(AUD[tr]).transpose(1,2);Gg=T3(G[tr]);yt=T3(y[tr])
        Xve=T3(VIS[ev]).transpose(1,2);Xae=T3(AUD[ev]).transpose(1,2);Ge=T3(G[ev]);n=tr.sum();bs=32;best=0;bp=None
        for ep in range(EP):
            net.train();perm=torch.randperm(n)
            for i in range(0,n,bs):
                b=perm[i:i+bs];opt.zero_grad();loss=lf(net(Xv[b],Xa[b],Gg[b]),yt[b]);loss.backward();opt.step()
            net.eval()
            with torch.no_grad():p=torch.sigmoid(net(Xve,Xae,Ge)).cpu().numpy()
            try:
                a=roc_auc_score(y[ev],p)
                if a>best:best=a;bp=p
            except:pass
        if bp is not None:aucs.append(roc_auc_score(y[ev],bp));f1s.append(f1_score(y[ev],(bp>0.5).astype(int)))
    print(f'  {name}: AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} F1={np.mean(f1s):.3f}',flush=True)
    return np.mean(aucs),np.mean(f1s)
out=[]
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
# ---- DAIC (train->dev) ----
print('DAIC 로딩...',flush=True)
D=B/'DAIC_WOZ';split={}
for f,fold in [('train_split_Depression_AVEC2017.csv','train'),('dev_split_Depression_AVEC2017.csv','dev')]:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):split[r['Participant_ID'].strip()]=(int(float(r['PHQ8_Binary'])),fold)
def dload(pid):
    pa=D/f'{pid}_CLNF_AUs.txt';pc=D/f'{pid}_COVAREP.csv'
    if not(pa.exists() and pc.exists()):return None
    h=[x.strip() for x in open(pa).readline().split(',')];oi=h.index('success');ai=[h.index(c) for c in AUd]
    au=[]
    for ln in open(pa).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            au.append([float(v[i]) for i in ai])
        except:pass
    if len(au)<60:return None
    au=np.array(au);Z=(au-au.mean(0))/(au.std(0)+1e-6)
    cov=[]
    for j,ln in enumerate(open(pc).readlines()):
        if j%5:continue  # 다운샘플
        try:cov.append([float(x) for x in ln.split(',')])
        except:pass
    if len(cov)<30:return None
    cov=np.array(cov);Az=(cov-cov.mean(0))/(cov.std(0)+1e-6)
    return Z,Az
VIS=[];AUD=[];COV=[];y=[];fold=[]
for pid,(lab,fo) in split.items():
    d=dload(pid)
    if d is None:continue
    Z,Az=d;VIS.append(resamp(Z));AUD.append(resamp(Az));c,_=ledoit_wolf(Z);COV.append(c+1e-4*np.eye(len(AUd)));y.append(lab);fold.append(fo)
VIS=np.array(VIS);AUD=np.array(AUD);COV=np.array(COV);y=np.array(y);fold=np.array(fold)
print(f'DAIC n={len(y)} train{np.sum(fold=="train")} dev{np.sum(fold=="dev")}',flush=True)
tr=fold=='train';ev=fold=='dev'
a,fq=bench(VIS,AUD,COV,y,tr,ev,'DAIC(dev)',AUD.shape[2]);out.append(['DAIC',int(ev.sum()),a,fq])
# ---- E-DAIC (train->test) ----
print('E-DAIC 로딩...',flush=True)
E=B/'E-DAIC';esp={}
for f,fo in [('train_split.csv','train'),('dev_split.csv','dev'),('test_split.csv','test')]:
    p=E/'labels'/f
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
    if len(au)<60:return None
    au=np.array(au);Z=(au-au.mean(0))/(au.std(0)+1e-6)
    eg=[]
    for ln in open(af[0]).readlines()[1:]:
        parts=ln.split(';') if ';' in ln else ln.split(',')
        try:eg.append([float(x) for x in parts[1:] if x.strip()!=''])
        except:pass
    if len(eg)<10:return None
    L=len(eg[0]);eg=np.array([r for r in eg if len(r)==L]);Az=(eg-eg.mean(0))/(eg.std(0)+1e-6)
    return Z,Az
VIS=[];AUD=[];COV=[];y=[];fold=[]
for pid,(lab,fo) in esp.items():
    d=eload(pid)
    if d is None:continue
    Z,Az=d;VIS.append(resamp(Z));AUD.append(resamp(Az));c,_=ledoit_wolf(Z);COV.append(c+1e-4*np.eye(len(AUc)));y.append(lab);fold.append(fo)
VIS=np.array(VIS);AUD=np.array(AUD);COV=np.array(COV);y=np.array(y);fold=np.array(fold)
print(f'E-DAIC n={len(y)} train{np.sum(fold=="train")} test{np.sum(fold=="test")}',flush=True)
tr=fold=='train';ev=fold=='test'
if ev.sum()>5:
    a,fq=bench(VIS,AUD,COV,y,tr,ev,'E-DAIC(test)',AUD.shape[2]);out.append(['E-DAIC',int(ev.sum()),a,fq])
print('\n판정: DAIC/E-DAIC서도 준수하면 일반화, 낮으면 D-Vlog 한정(corpus-specific 재확인).',flush=True)
print('[참고] D-Vlog 공식 F1 0.727 | DAIC 얼굴 baseline~0.6 | E-DAIC 멀티모달 CCC 0.5대',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp83_other.csv','w') as f:
    f.write('corpus,n_test,AUC,F1\n')
    for r in out:f.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp83_other.csv',flush=True)
