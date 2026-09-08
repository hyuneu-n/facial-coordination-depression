"""exp91b — coordination 히트맵 개선. 한글 라벨 + 차이패널 top 쌍 박스. top 쌍 콘솔 출력."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
import numpy as np, glob, re, os
import pandas as pd
# 한글 폰트 시도 (없으면 영문)
KOR=None
for c in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.exists(c):
        fm.fontManager.addfont(c); KOR=fm.FontProperties(fname=c).get_name(); break
if KOR: plt.rcParams['font.family']=KOR
plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size':11})
ROOT='/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted/'
FD='/home/hyuneun/disk_b/🟡facial-prodrome/figs/'
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r',
     'AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
LBL=[a.replace('_r','') for a in AUS]
REGION={'AU01':'이마','AU02':'이마','AU04':'미간','AU05':'눈','AU06':'볼','AU07':'눈','AU09':'코',
        'AU10':'윗입술','AU12':'입꼬리','AU14':'입','AU15':'입꼬리','AU17':'턱','AU20':'입',
        'AU23':'입술','AU25':'입','AU26':'턱','AU45':'눈깜빡'}
def subj_corr(folder):
    mats=[]
    for f in glob.glob(folder+'/Q*.csv'):
        if not re.match(r'^Q\d+\.csv$',os.path.basename(f)):continue
        try:df=pd.read_csv(f)
        except:continue
        df.columns=[c.strip() for c in df.columns]
        if 'success' not in df or not set(AUS).issubset(df.columns):continue
        df=df[df['success']==1]
        if 'confidence' in df:df=df[df['confidence']>0.9]
        if len(df)<30:continue
        X=df[AUS].values.astype(float);X=(X-X.mean(0))/(X.std(0)+1e-6);mats.append(X)
    if not mats:return None
    return np.corrcoef(np.vstack(mats).T)
HC=[];MDD=[]
for d in sorted(glob.glob(ROOT+'*/')):
    name=os.path.basename(d.rstrip('/'));c=subj_corr(d)
    if c is None or np.isnan(c).any():continue
    (HC if name.startswith('HC') else MDD).append(c)
HC=np.array(HC);MDD=np.array(MDD);mHC=HC.mean(0);mMDD=MDD.mean(0);diff=mMDD-mHC
# top +/- off-diagonal pairs
n=len(LBL);pairs=[]
for i in range(n):
    for j in range(i+1,n):
        pairs.append((diff[i,j],i,j))
pairs.sort(reverse=True)
TOPP=pairs[:4]; TOPN=pairs[-4:]
print('▲ 우울에서 결합↑ (빨강) top4:')
for v,i,j in TOPP: print(f'   {LBL[i]}({REGION[LBL[i]]})–{LBL[j]}({REGION[LBL[j]]})  +{v:.2f}')
print('▼ 우울에서 결합↓ (파랑) top4:')
for v,i,j in TOPN: print(f'   {LBL[i]}({REGION[LBL[i]]})–{LBL[j]}({REGION[LBL[j]]})  {v:.2f}')
# plot
t_hc='건강군 (HC)' if KOR else 'HC (healthy)'
t_mdd='우울군 (MDD)' if KOR else 'MDD (depressed)'
t_df='차이 = 우울군 - 건강군' if KOR else 'Difference (MDD - HC)'
sup='얼굴 근육(AU) 간 coordination 구조 — CMDC' if KOR else 'Inter-AU coordination (CMDC)'
fig,axs=plt.subplots(1,3,figsize=(16,5.4))
for ax,M,t,cm,vlim in [(axs[0],mHC,t_hc,'viridis',(-.3,1)),(axs[1],mMDD,t_mdd,'viridis',(-.3,1)),
                        (axs[2],diff,t_df,'coolwarm',(-.3,.3))]:
    im=ax.imshow(M,cmap=cm,vmin=vlim[0],vmax=vlim[1])
    ax.set_xticks(range(n));ax.set_xticklabels(LBL,rotation=90,fontsize=8)
    ax.set_yticks(range(n));ax.set_yticklabels(LBL,fontsize=8)
    ax.set_title(t,fontsize=13);plt.colorbar(im,ax=ax,fraction=0.046)
# 차이 패널에 top+ 박스
for v,i,j in TOPP:
    for (a,b) in [(i,j),(j,i)]:
        axs[2].add_patch(mpatches.Rectangle((b-0.5,a-0.5),1,1,fill=False,edgecolor='black',lw=2.2))
fig.suptitle(sup,fontsize=14)
plt.tight_layout();plt.savefig(FD+'fig5_coordination.png',dpi=140,bbox_inches='tight');plt.close()
print('KOR font:',KOR);print('DONE fig5_coordination.png')
