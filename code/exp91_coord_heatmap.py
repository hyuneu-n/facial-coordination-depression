"""exp91 — coordination 히트맵. CMDC AU 상관행렬 HC vs MDD vs 차이. 실제 데이터."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, glob, re, os
import pandas as pd
plt.rcParams.update({'font.size':11})
ROOT='/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted/'
FD='/home/hyuneun/disk_b/🟡facial-prodrome/figs/'
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r',
     'AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
LBL=[a.replace('_r','') for a in AUS]
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
        X=df[AUS].values.astype(float)
        X=(X-X.mean(0))/(X.std(0)+1e-6)
        mats.append(X)
    if not mats:return None
    X=np.vstack(mats)
    return np.corrcoef(X.T)
HC=[];MDD=[]
for d in sorted(glob.glob(ROOT+'*/')):
    name=os.path.basename(d.rstrip('/'))
    c=subj_corr(d)
    if c is None or np.isnan(c).any():continue
    (HC if name.startswith('HC') else MDD).append(c)
HC=np.array(HC);MDD=np.array(MDD)
print(f'HC {len(HC)} subj, MDD {len(MDD)} subj',flush=True)
mHC=HC.mean(0);mMDD=MDD.mean(0);diff=mMDD-mHC
fig,axs=plt.subplots(1,3,figsize=(16,5.2))
for ax,M,t,cm,vlim in [(axs[0],mHC,'HC (healthy)','viridis',(-.3,1)),
                        (axs[1],mMDD,'MDD (depressed)','viridis',(-.3,1)),
                        (axs[2],diff,'Difference (MDD - HC)','coolwarm',(-.3,.3))]:
    im=ax.imshow(M,cmap=cm,vmin=vlim[0],vmax=vlim[1])
    ax.set_xticks(range(len(LBL)));ax.set_xticklabels(LBL,rotation=90,fontsize=8)
    ax.set_yticks(range(len(LBL)));ax.set_yticklabels(LBL,fontsize=8)
    ax.set_title(t,fontsize=13);plt.colorbar(im,ax=ax,fraction=0.046)
fig.suptitle('Inter-AU coordination structure (CMDC) — covariance/correlation on the SPD manifold',fontsize=14)
plt.tight_layout();plt.savefig(FD+'fig5_coordination.png',dpi=140,bbox_inches='tight');plt.close()
# 요약 수치: 평균 off-diagonal coordination 강도
def offmean(M):
    m=~np.eye(len(M),dtype=bool);return np.abs(M[m]).mean()
print(f'평균 |coordination| HC={offmean(mHC):.3f} MDD={offmean(mMDD):.3f}',flush=True)
print('DONE fig5_coordination.png',flush=True)
