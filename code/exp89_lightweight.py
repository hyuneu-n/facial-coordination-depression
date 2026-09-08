"""
exp89 — 최종 모델 경량성 정량화 (엣지 논거). 파라미터 수 + 추론속도(CPU/GPU).
최종 = SPD-manifold(visual) + audio-CNN + concat. Transformer(exp78)와 비교.
결과: results/exp89_lightweight.csv
"""
import numpy as np, warnings, glob, time, csv
from pathlib import Path
import torch, torch.nn as nn
warnings.filterwarnings('ignore')
T=256;W=64;STR=32;K=len(range(0,T-W+1,STR))  # window 수
# 오디오 차원 확인
ap=glob.glob('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog/*/*_acoustic.npy')
Ca=np.load(ap[0]).shape[1] if ap else 25
d=20
class SPDNet(nn.Module):
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
        Bs,Kk,dd,_=covs.shape;v=self.sl(covs.reshape(Bs*Kk,dd,dd)).reshape(Bs,Kk,-1);return self.pre(v).mean(1)
class CNNb(nn.Module):
    def __init__(self,C):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(C,48,5,padding=2),nn.BatchNorm1d(48),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(48,96,5,padding=2),nn.BatchNorm1d(96),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
    def forward(self,x):return self.net(x).squeeze(-1)
class Final(nn.Module):
    def __init__(self,d,Ca):
        super().__init__();self.spd=SPDNet(d);self.a=CNNb(Ca)
        self.head=nn.Sequential(nn.Linear(64+96,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,1))
    def forward(self,covs,xa):return self.head(torch.cat([self.spd(covs),self.a(xa)],1)).squeeze(-1)
import math
class Transf(nn.Module):  # 비교용(exp78류)
    def __init__(self,C):
        super().__init__();self.proj=nn.Linear(C,128)
        enc=nn.TransformerEncoderLayer(128,4,256,0.3,batch_first=True);self.tr=nn.TransformerEncoder(enc,2)
        self.head=nn.Linear(128,1)
    def forward(self,x):h=self.tr(self.proj(x));return self.head(h.mean(1)).squeeze(-1)
def nparams(m):return sum(p.numel() for p in m.parameters())
final=Final(d,Ca);tr=Transf(20)
pf=nparams(final);pt=nparams(tr)
print(f'오디오 차원 Ca={Ca}, window 수 K={K}',flush=True)
print(f'최종(SPD+audio+concat) 파라미터: {pf:,} ({pf/1e3:.1f}K)',flush=True)
print(f'Transformer(비교) 파라미터: {pt:,} ({pt/1e3:.1f}K)',flush=True)
# 추론속도 (per sample)
def bench(dev):
    final.to(dev).eval()
    covs=torch.randn(1,K,d,d,device=dev);covs=covs@covs.transpose(2,3)+1e-2*torch.eye(d,device=dev)
    xa=torch.randn(1,Ca,T,device=dev)
    with torch.no_grad():
        for _ in range(5):final(covs,xa)  # warmup
        if dev=='cuda':torch.cuda.synchronize()
        t0=time.time()
        for _ in range(50):final(covs,xa)
        if dev=='cuda':torch.cuda.synchronize()
        return (time.time()-t0)/50*1000
cpu_ms=bench('cpu')
gpu_ms=bench('cuda') if torch.cuda.is_available() else float('nan')
print(f'추론 지연(1 sample): CPU {cpu_ms:.1f} ms | GPU {gpu_ms:.2f} ms',flush=True)
print(f'[비교] 전형적 딥 멀티모달 SOTA = 수~수십 M 파라미터 (우리는 {pf/1e3:.0f}K = 수백배 작음)',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp89_lightweight.csv','w') as f:
    f.write('metric,value\n')
    f.write(f'params_final,{pf}\nparams_K,{pf/1e3:.1f}\nparams_transformer,{pt}\ncpu_ms,{cpu_ms:.2f}\ngpu_ms,{gpu_ms:.3f}\naudio_dim,{Ca}\n')
print('DONE → exp89_lightweight.csv',flush=True)
