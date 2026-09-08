"""exp109 — 협응 부위 심화. 어느 AU 쌍의 협응이 우울 판별을 주도하나? 임상(CMDC) vs vlog(LMVD).
각 AU쌍 상관값의 단일-쌍 판별력(|AUC-0.5|) 히트맵 + top쌍 + 두 코퍼스 공통쌍. fig12.
"""
import numpy as np, warnings, os, re, glob
import pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
LBL=[a.replace('_r','') for a in AUS];D=len(AUS)
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/'
REG={'AU01':'이마','AU02':'이마','AU04':'미간','AU05':'눈','AU06':'볼','AU07':'눈','AU09':'코','AU10':'윗입술','AU12':'입꼬리','AU14':'입','AU15':'입꼬리','AU17':'턱','AU20':'입','AU23':'입술','AU25':'입','AU26':'턱','AU45':'눈'}
def persec(f):
    try:df=pd.read_csv(f,usecols=lambda c:c.strip() in set(['timestamp','success','confidence']+AUS))
    except:return None
    df.columns=[c.strip() for c in df.columns]
    if 'timestamp' not in df or not set(AUS).issubset(df.columns):return None
    if 'success' in df:df=df[df['success']==1]
    if len(df)<50:return None
    df=df.copy();df['sec']=np.floor(df['timestamp']).astype(int)
    return df.groupby('sec')[AUS].mean().values
def corr(series):
    Z=(series-series.mean(0))/(series.std(0)+1e-6);return np.corrcoef(Z.T)
def disc_map(CORRS,Y):
    Y=np.array(Y);M=np.zeros((D,D))
    for i in range(D):
        for j in range(i+1,D):
            v=np.array([c[i,j] for c in CORRS])
            try:a=roc_auc_score(Y,v)
            except:a=0.5
            M[i,j]=M[j,i]=abs(a-0.5)  # 판별력
    return M
# CMDC
def load_cmdc():
    C=[];Y=[]
    for dfold in sorted(glob.glob(DATA+'CMDC/extracted/*/')):
        name=os.path.basename(dfold.rstrip('/'))
        if not(name.startswith('HC') or name.startswith('MDD')):continue
        segs=[persec(f) for f in glob.glob(dfold+'Q*.csv') if re.match(r'^Q\d+\.csv$',os.path.basename(f))]
        segs=[s for s in segs if s is not None]
        if not segs:continue
        s=np.vstack(segs)
        if len(s)<40:continue
        cc=corr(s)
        if np.isnan(cc).any():continue
        C.append(cc);Y.append(1 if name.startswith('MDD') else 0)
    return C,Y
def load_lmvd():
    def lm(i):return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
    C=[];Y=[]
    for f in sorted(glob.glob(DATA+'LMVD/extracted/Video_feature/*.csv')):
        m=re.search(r'(\d+)',os.path.basename(f))
        if not m:continue
        ps=persec(f)
        if ps is None or len(ps)<40:continue
        cc=corr(ps)
        if np.isnan(cc).any():continue
        C.append(cc);Y.append(lm(int(m.group(1))))
    return C,Y
Cc,Yc=load_cmdc();Mc=disc_map(Cc,Yc);print(f'CMDC n={len(Yc)}',flush=True)
Cl,Yl=load_lmvd();Ml=disc_map(Cl,Yl);print(f'LMVD n={len(Yl)}',flush=True)
def top(M,k=6):
    idx=[(M[i,j],i,j) for i in range(D) for j in range(i+1,D)];idx.sort(reverse=True)
    return idx[:k]
print('=== CMDC(임상) top 판별 협응쌍 ===',flush=True)
tc=top(Mc)
for v,i,j in tc:print(f'  {LBL[i]}({REG[LBL[i]]})-{LBL[j]}({REG[LBL[j]]}) |AUC-.5|={v:.3f}',flush=True)
print('=== LMVD(vlog) top 판별 협응쌍 ===',flush=True)
tl=top(Ml)
for v,i,j in tl:print(f'  {LBL[i]}({REG[LBL[i]]})-{LBL[j]}({REG[LBL[j]]}) |AUC-.5|={v:.3f}',flush=True)
sc=set((i,j) for _,i,j in tc);sl=set((i,j) for _,i,j in tl)
print(f'공통 top쌍: {len(sc&sl)}/{len(sc)}',flush=True)
# fig12
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib import font_manager as fm
for c2 in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c2):fm.fontManager.addfont(c2);plt.rcParams['font.family']=fm.FontProperties(fname=c2).get_name()
plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(1,2,figsize=(14,6))
for ax,M,t,tp in [(axs[0],Mc,'CMDC (임상)',tc),(axs[1],Ml,'LMVD (vlog)',tl)]:
    im=ax.imshow(M,cmap='Reds',vmin=0,vmax=max(Mc.max(),Ml.max()))
    ax.set_xticks(range(D));ax.set_xticklabels(LBL,rotation=90,fontsize=8)
    ax.set_yticks(range(D));ax.set_yticklabels(LBL,fontsize=8)
    for v,i,j in tp:
        for a,b in [(i,j),(j,i)]:ax.add_patch(mp.Rectangle((b-.5,a-.5),1,1,fill=False,edgecolor='#1565C0',lw=2))
    ax.set_title(f'{t} — 협응쌍 판별력 |AUC-0.5|\n(파란박스=top6)',fontsize=12);plt.colorbar(im,ax=ax,fraction=0.046)
fig.suptitle('어느 AU 쌍의 협응이 우울 판별을 주도하나 — 임상 vs vlog',fontsize=14)
plt.tight_layout();plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig12_region.png',dpi=140,bbox_inches='tight')
print('DONE fig12_region.png',flush=True)
