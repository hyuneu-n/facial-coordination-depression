"""exp90 — 발표용 그림 3종 (영문 라벨, 깔끔). 모두 실제 실험 수치."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
FD='/home/hyuneun/disk_b/🟡facial-prodrome/figs/'
plt.rcParams.update({'font.size':13,'axes.edgecolor':'#333'})

# Fig1 — cross-corpus AUC (final model)
corp=['CMDC','D-Vlog','LMVD','E-DAIC'];auc=[0.89,0.808,0.627,0.624]
col=['#2E7D32','#2E7D32','#9E9E9E','#9E9E9E']  # strong green / weak gray
fig,ax=plt.subplots(figsize=(6.2,4))
b=ax.bar(corp,auc,color=col,width=0.62)
ax.axhline(0.5,ls='--',c='#c0392b',lw=1.2);ax.text(3.35,0.51,'chance',color='#c0392b',fontsize=10)
for r,v in zip(b,auc):ax.text(r.get_x()+r.get_width()/2,v+0.01,f'{v:.2f}',ha='center',fontweight='bold')
ax.set_ylim(0,1.0);ax.set_ylabel('AUC');ax.set_title('Cross-corpus performance (final model)')
ax.text(0.5,0.94,'strong',color='#2E7D32',fontsize=10,ha='center',transform=ax.get_xaxis_transform())
ax.text(2.5,0.70,'weak',color='#9E9E9E',fontsize=10,ha='center',transform=ax.get_xaxis_transform())
plt.tight_layout();plt.savefig(FD+'fig1_crosscorpus.png',dpi=150,bbox_inches='tight');plt.close()

# Fig2 — lightweight (params, log scale)
mods=['Ours\n(SPD+audio)','Transformer\n(baseline)','Typical deep\nmultimodal SOTA']
par=[48929,267777,5000000];labs=['49K','268K','~5M'];colr=['#1565C0','#90A4AE','#B0BEC5']
order=[2,1,0]  # bottom→top: Ours ends on top
fig,ax=plt.subplots(figsize=(7.2,3.9))
b=ax.barh([mods[i] for i in order],[par[i] for i in order],color=[colr[i] for i in order]);ax.set_xscale('log')
for k,i in enumerate(order):ax.text(par[i]*1.18,k,labs[i],va='center',fontweight='bold')
ax.set_xlabel('# parameters (log scale)')
ax.set_title('Model size: ~100x smaller than deep SOTA\n(CPU 1.6 ms / sample — on-device)',fontsize=13)
ax.set_xlim(2e4,3e7)
plt.tight_layout();plt.savefig(FD+'fig2_lightweight.png',dpi=150,bbox_inches='tight');plt.close()

# Fig3 — D-Vlog official benchmark. 원논문은 F1만 보고(AUC 미보고) → 정직 표기
x=np.arange(2);w=0.35
orig=[0.635,np.nan]  # F1=0.635(보고), AUC=미보고
ours=[0.794,0.807]   # F1, AUC
fig,ax=plt.subplots(figsize=(6.0,4.2))
b1=ax.bar(x-w/2,np.nan_to_num(orig,nan=0),w,label='D-Vlog original (Yoon 2022)',color='#B0BEC5')
b2=ax.bar(x+w/2,ours,w,label='Ours',color='#1565C0')
ax.text(0-w/2,0.635+0.01,'0.64',ha='center',fontsize=11,fontweight='bold')
ax.text(1-w/2,0.03,'not reported\nin original',ha='center',fontsize=9,color='#6b7280')
for r,val in zip(b2,ours):ax.text(r.get_x()+r.get_width()/2,val+0.01,f'{val:.2f}',ha='center',fontsize=11,fontweight='bold')
ax.set_xticks(x);ax.set_xticklabels(['F1','AUC']);ax.set_ylim(0,1.0)
ax.set_title('D-Vlog official split (vs original)');ax.legend(fontsize=10,loc='upper left')
plt.tight_layout();plt.savefig(FD+'fig3_dvlog.png',dpi=150,bbox_inches='tight');plt.close()
print('DONE figs: fig1_crosscorpus fig2_lightweight fig3_dvlog',flush=True)
