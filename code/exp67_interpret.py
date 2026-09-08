"""
exp67 — 해석(논문 Figure 재료): 어느 AU/얼굴부위 expressivity가 우울 판별하나 + 기전 효과크기.
LMVD: AU별 tension(평균, pop-표준화) MDD vs HC + Cliff's delta 랭킹.
D-Vlog: 얼굴부위(brow/eye/nose/mouth)별 움직임 활성도 MDD vs HC.
그림: figs/expressivity_mechanism.png (그룹 막대 + 효과크기). 결과: results/exp67_interpret.csv
"""
import numpy as np, warnings, csv, glob
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
AUc=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45']
AUcR=[a+'_r' for a in AUc]
def cliffs(a,b):
    a=np.asarray(a);b=np.asarray(b);gt=(a[:,None]>b[None,:]).sum();lt=(a[:,None]<b[None,:]).sum();return (gt-lt)/(len(a)*len(b))
# ---- LMVD: AU별 tension ----
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423): return 1
    if (602<=i<=1116) or (1425<=i<=1824): return 0
    return None
M=[];yL=[]
for f in sorted(glob.glob(str(V/'*.csv'))):
    l=lab(int(Path(f).stem))
    if l is None:continue
    h=[x.strip() for x in open(f).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUcR]
    except:continue
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    if len(fe)>=60: M.append(np.nan_to_num(np.array(fe)).mean(0));yL.append(l)  # subject별 평균 AU(tension)
M=np.array(M);yL=np.array(yL)
# pop 표준화
Ms=(M-M.mean(0))/(M.std(0)+1e-6)
print(f'LMVD n={len(yL)} — AU별 tension(평균활성) 효과크기(우울-정상):',flush=True)
rows=[]
for j,au in enumerate(AUc):
    d=cliffs(Ms[yL==1,j],Ms[yL==0,j]); rows.append((au,M[yL==1,j].mean(),M[yL==0,j].mean(),d))
for au,md,hc,d in sorted(rows,key=lambda x:x[3])[:8]:
    print(f'  {au}: MDD={md:.3f} HC={hc:.3f} cliffδ={d:+.3f}',flush=True)
# ---- 그림: 상위 판별 AU tension 그룹평균 ----
rows_sorted=sorted(rows,key=lambda x:abs(x[3]),reverse=True)[:10]
aus=[r[0] for r in rows_sorted]; mdv=[r[1] for r in rows_sorted]; hcv=[r[2] for r in rows_sorted]
figd=Path('/Users/nonexist')  # placeholder
outfig=B.parent/'figs'/'expressivity_mechanism.png'
outfig.parent.mkdir(exist_ok=True)
fig,ax=plt.subplots(1,2,figsize=(13,5))
x=np.arange(len(aus));w=0.38
ax[0].bar(x-w/2,hcv,w,label='HC',color='#4472C4');ax[0].bar(x+w/2,mdv,w,label='MDD',color='#C00000')
ax[0].set_xticks(x);ax[0].set_xticklabels(aus,rotation=45);ax[0].set_ylabel('mean AU intensity (tension)')
ax[0].set_title('LMVD: facial expressivity (tension) by AU — MDD lower');ax[0].legend()
# D-Vlog 부위별
R=B/'D-Vlog'
def nf(Vv):
    T=Vv.shape[0];P=Vv.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,68,2)
REG={'brow(17-26)':range(17,27),'eye(36-47)':range(36,48),'nose(27-35)':range(27,36),'mouth(48-67)':range(48,68),'jaw(0-16)':range(0,17)}
regmv={k:[[],[]] for k in REG}
nd=0
for r_ in csv.DictReader(open(R/'labels.csv')):
    idx=r_['index'].strip();f=R/idx/f'{idx}_visual.npy'
    if not f.exists():continue
    try:Vv=np.load(f)
    except:continue
    if Vv.ndim!=2 or Vv.shape[0]<60 or Vv.shape[1]!=136:continue
    P=nf(Vv);lb=1 if r_['label'].strip().lower().startswith('depress') else 0
    vel=np.linalg.norm(np.diff(P,axis=0),axis=2)  # (T-1,68) 점별 속도
    for k,rg in REG.items(): regmv[k][lb].append(vel[:,list(rg)].mean())
    nd+=1
print(f'\nD-Vlog n={nd} — 부위별 움직임 활성도(우울 vs 정상):',flush=True)
regs=list(REG);md_r=[];hc_r=[];drow=[]
for k in regs:
    md=np.mean(regmv[k][1]);hc=np.mean(regmv[k][0]);d=cliffs(regmv[k][1],regmv[k][0])
    md_r.append(md);hc_r.append(hc);drow.append((k,md,hc,d))
    print(f'  {k}: MDD={md:.3f} HC={hc:.3f} cliffδ={d:+.3f}',flush=True)
x2=np.arange(len(regs))
ax[1].bar(x2-w/2,hc_r,w,label='HC',color='#4472C4');ax[1].bar(x2+w/2,md_r,w,label='MDD',color='#C00000')
ax[1].set_xticks(x2);ax[1].set_xticklabels(regs,rotation=30,ha='right');ax[1].set_ylabel('movement amplitude')
ax[1].set_title('D-Vlog: facial region movement — MDD lower');ax[1].legend()
plt.tight_layout();plt.savefig(outfig,dpi=130);print(f'\n그림 저장: {outfig}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp67_interpret.csv','w') as f:
    f.write('dataset,unit,MDD,HC,cliffs_delta\n')
    for au,md,hc,d in rows: f.write(f'LMVD,{au},{md:.4f},{hc:.4f},{d:.4f}\n')
    for k,md,hc,d in drow: f.write(f'D-Vlog,{k},{md:.4f},{hc:.4f},{d:.4f}\n')
print('DONE → exp67_interpret.csv + figs/expressivity_mechanism.png',flush=True)
