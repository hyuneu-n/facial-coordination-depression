"""exp110 — GRU 포함 최종모델 경량 재측정. SPD+GRU+audio+concat. param·CPU/GPU 속도.
exp89(GRU없음 48.9K)와 비교. 배포 논거 갱신.
"""
import numpy as np, warnings, glob, time
import torch, torch.nn as nn
warnings.filterwarnings('ignore')
T=256;W=64;STR=32;K=len(range(0,T-W+1,STR));d=20
ap=glob.glob('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog/*/*_acoustic.npy')
Ca=np.load(ap[0]).shape[1] if ap else 25
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
        Bs,Kk,dd,_=covs.shape;v=self.sl(covs.reshape(Bs*Kk,dd,dd)).reshape(Bs,Kk,-1);return self.pre(v)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Final(nn.Module):
    def __init__(self,d,Ca):
        super().__init__();self.spd=SPDseq(d);self.gru=nn.GRU(64,32,batch_first=True,bidirectional=True)
        self.a=CNNb(Ca);self.cls=nn.Sequential(nn.Linear(64+96,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,xa):o,_=self.gru(self.spd(covs));return self.cls(torch.cat([o.mean(1),self.a(xa)],1)).squeeze(-1)
def npar(m):return sum(p.numel() for p in m.parameters())
net=Final(d,Ca)
pf=npar(net);pg=sum(p.numel() for p in net.gru.parameters())
print(f'최종(SPD+GRU+audio+concat) 파라미터: {pf:,} ({pf/1e3:.1f}K)  [그중 GRU={pg:,}]',flush=True)
print(f'참고 exp89(GRU없음)=48,929. 증가분={pf-48929:,}',flush=True)
def bench(dev):
    net.to(dev).eval()
    covs=torch.randn(1,K,d,d,device=dev);covs=covs@covs.transpose(2,3)+1e-2*torch.eye(d,device=dev)
    xa=torch.randn(1,Ca,T,device=dev)
    with torch.no_grad():
        for _ in range(5):net(covs,xa)
        if dev=='cuda':torch.cuda.synchronize()
        t0=time.time()
        for _ in range(50):net(covs,xa)
        if dev=='cuda':torch.cuda.synchronize()
        return (time.time()-t0)/50*1000
print(f'추론(1샘플): CPU {bench("cpu"):.2f} ms | GPU {bench("cuda") if torch.cuda.is_available() else float("nan"):.2f} ms',flush=True)
print(f'[비교] Transformer 268K, 딥 멀티모달 SOTA 수~수십M. 우리 {pf/1e3:.0f}K = 여전히 수십~수백배 작음',flush=True)
print('DONE',flush=True)
