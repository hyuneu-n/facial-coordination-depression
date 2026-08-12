"""
발표/논문용 그래프 (교수님 HIS/Mifu-ER 스타일: 깔끔한 막대+에러바+유의표시).
G1 데이터별 최종 AUC / G2 앵커효과 / G3 협응5종 / G4 모델5종 / G5 판별 얼굴
"""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
OUT='/home/hyuneun/disk_b/🟡facial-prodrome/results/'
NAVY='#1f3b6f'; GRAY='#b0b0b0'; RED='#c0392b'; GREEN='#27ae60'

# G1 — 데이터별 최종 AUC (chance line, 유의표시)
fig,ax=plt.subplots(figsize=(5,4))
d=['CMDC\n(anchor)','DAIC\n(anchor)','E-DAIC\n(quality gate)']; auc=[0.889,0.711,0.563]; err=[0.022,0.046,0.027]; sig=['**','*','']
cols=[NAVY,NAVY,GRAY]
b=ax.bar(d,auc,yerr=err,color=cols,capsize=4,width=0.6,edgecolor='k',linewidth=0.5)
ax.axhline(0.5,ls='--',c='gray',lw=1); ax.text(2.3,0.51,'chance',fontsize=9,color='gray')
for i,(a,s) in enumerate(zip(auc,sig)): ax.text(i,a+err[i]+0.015,f'{a:.3f}{s}',ha='center',fontweight='bold')
ax.set_ylabel('AUC (depression detection)'); ax.set_ylim(0.4,1.0)
ax.set_title('Anchored inter-AU coupling across datasets')
plt.tight_layout(); plt.savefig(OUT+'G1_datasets.png'); plt.close()

# G2 — 앵커 효과 (전체 vs 앵커)
fig,ax=plt.subplots(figsize=(5.5,4))
x=np.arange(2); w=0.35
whole=[0.40,0.76]; anchor=[0.889,0.711]  # [CMDC 전체는 exp1 0.40? CMDC전체=0.76, DAIC전체=0.40]
# 정확히: CMDC 전체(all Q)=0.76, 앵커=0.889 / DAIC 전체=0.40, 앵커=0.711
whole=[0.828,0.600]; anchor=[0.889,0.711]
ax.bar(x-w/2,whole,w,label='Whole interview',color=GRAY,edgecolor='k',lw=0.5)
ax.bar(x+w/2,anchor,w,label='Symptom anchor',color=NAVY,edgecolor='k',lw=0.5)
for i in range(2):
    ax.annotate('',xy=(i+w/2,anchor[i]+0.03),xytext=(i-w/2,whole[i]+0.03),arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.5))
    ax.text(i,max(anchor[i],whole[i])+0.06,f'+{anchor[i]-whole[i]:.2f}',ha='center',color=GREEN,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['CMDC','DAIC']); ax.axhline(0.5,ls='--',c='gray',lw=1)
ax.set_ylabel('AUC'); ax.set_ylim(0.3,1.0); ax.legend(); ax.set_title('Effect of question anchoring')
plt.tight_layout(); plt.savefig(OUT+'G2_anchor.png'); plt.close()

# G3 — 협응 측정 5종 (CMDC)
fig,ax=plt.subplots(figsize=(6,4))
m=['Covariance\n(coupling)','Lagged corr\n(timing)','NMF\nsynergy','HMM\nstate','MdRQA\nrigidity']
v=[0.885,0.802,0.535,0.441,0.364]; cc=[NAVY,NAVY,GRAY,GRAY,GRAY]
ax.bar(m,v,color=cc,edgecolor='k',lw=0.5,width=0.65)
for i,a in enumerate(v): ax.text(i,a+0.015,f'{a:.2f}',ha='center',fontsize=9)
ax.axhline(0.5,ls='--',c='gray',lw=1); ax.set_ylabel('AUC'); ax.set_ylim(0.3,1.0)
ax.set_title('Coupling measures: pairwise coupling is diagnostic')
plt.tight_layout(); plt.savefig(OUT+'G3_coordination.png'); plt.close()

# G4 — 모델 5종 (CMDC)
fig,ax=plt.subplots(figsize=(6,4))
m=['Tangent+\nLogReg','SPDNet','VNN','Transformer','Sparse\nL1']; v=[0.885,0.79,0.55,0.27,0.75]
cc=[NAVY,GRAY,GRAY,RED,GRAY]
ax.bar(m,v,color=cc,edgecolor='k',lw=0.5,width=0.65)
for i,a in enumerate(v): ax.text(i,a+0.015,f'{a:.2f}',ha='center',fontsize=9)
ax.axhline(0.5,ls='--',c='gray',lw=1); ax.set_ylabel('AUC'); ax.set_ylim(0.2,1.0)
ax.set_title('Lightweight geometry beats heavy models (n=44)')
plt.tight_layout(); plt.savefig(OUT+'G4_models.png'); plt.close()
print('G1-G4 saved')
