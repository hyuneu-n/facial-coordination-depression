"""exp115 — 교차모달 조기성. 얼굴-음성 연동 약화가 세션 어느 구간부터 나타나나?
세션을 5구간으로 나눠 각 구간의 face-voice coupling(얼굴 활동량-음성 활동량 상관) MDD vs HC.
연동 약화가 앞구간부터면 기전+조기성 통합. D-Vlog.
"""
import numpy as np, warnings, csv
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
NSEG=5
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
def fv_coupling(vis,aud):
    # 얼굴 활동량(프레임간 landmark 이동) vs 음성 활동량(에너지=행 norm), 구간별 상관
    n=min(len(vis),len(aud))
    if n<NSEG*10:return None
    vis=vis[:n];aud=aud[:n]
    fmov=np.concatenate([[0],np.linalg.norm(np.diff(vis,axis=0),axis=1)])  # 얼굴 움직임
    aeng=np.linalg.norm(aud,axis=1)  # 음성 에너지
    segs=[]
    for s in range(NSEG):
        i0=s*n//NSEG;i1=(s+1)*n//NSEG
        f=fmov[i0:i1];a=aeng[i0:i1]
        if len(f)<8 or f.std()<1e-8 or a.std()<1e-8:segs.append(np.nan);continue
        segs.append(abs(np.corrcoef(f,a)[0,1]))  # 얼굴-음성 연동 강도
    return segs
rows=list(csv.DictReader(open(R/'labels.csv')))
Seg=[];Y=[]
for r in rows:
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
    if not(fv.exists() and fa.exists()):continue
    try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
    except:continue
    if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136 or Au.ndim!=2:continue
    s=fv_coupling(nf(V),Au)
    if s is None or np.isnan(s).any():continue
    Seg.append(s);Y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
Seg=np.array(Seg);Y=np.array(Y)
print(f'n={len(Y)} (MDD={Y.sum()})',flush=True)
print('=== 구간별 얼굴-음성 연동강도 (MDD vs HC) ===',flush=True)
for s in range(NSEG):
    a=Seg[Y==1,s];b=Seg[Y==0,s]
    try:_,p=mannwhitneyu(a,b,alternative='two-sided')
    except:p=1
    pct=f'{s*20}-{(s+1)*20}%'
    dr='MDD<HC(약화)' if a.mean()<b.mean() else 'MDD>HC'
    print(f'  구간{pct:8s}: MDD={a.mean():.3f} HC={b.mean():.3f} [{dr}] p={p:.2e}',flush=True)
# fig14
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os
for c in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c):fm.fontManager.addfont(c);plt.rcParams['font.family']=fm.FontProperties(fname=c).get_name()
plt.rcParams['axes.unicode_minus']=False
x=[s*20+10 for s in range(NSEG)]
mM=[Seg[Y==1,s].mean() for s in range(NSEG)];seM=[Seg[Y==1,s].std()/np.sqrt((Y==1).sum()) for s in range(NSEG)]
mH=[Seg[Y==0,s].mean() for s in range(NSEG)];seH=[Seg[Y==0,s].std()/np.sqrt((Y==0).sum()) for s in range(NSEG)]
plt.figure(figsize=(6.4,4.4))
plt.errorbar(x,mH,yerr=seH,marker='o',color='#1565C0',lw=2.2,capsize=3,label='건강군 (HC)')
plt.errorbar(x,mM,yerr=seM,marker='s',color='#c0392b',lw=2.2,capsize=3,label='우울군 (MDD)')
plt.xlabel('세션 구간 (%)');plt.ylabel('얼굴-음성 연동 강도')
plt.title('얼굴-음성 연동 약화가 세션 어디부터 나타나나 — D-Vlog\n(기전 × 조기성 결합)',fontsize=12)
plt.legend(fontsize=10);plt.tight_layout()
plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig14_crossmodal_early.png',dpi=145,bbox_inches='tight')
print('DONE fig14',flush=True)
