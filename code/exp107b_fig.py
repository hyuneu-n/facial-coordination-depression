"""fig10 — 개별 vs 협응 vs 결합, 4코퍼스 (exp107 실측)."""
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np, os
for c in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c):fm.fontManager.addfont(c);plt.rcParams['font.family']=fm.FontProperties(fname=c).get_name()
plt.rcParams['axes.unicode_minus']=False;plt.rcParams.update({'font.size':12})
corp=['CMDC\n(임상)','E-DAIC\n(임상)','D-Vlog\n(vlog)','LMVD\n(vlog)']
ind=[0.807,0.571,0.677,0.754];coo=[0.908,0.621,0.667,0.737];both=[0.899,0.641,0.694,0.770]
x=np.arange(len(corp));w=0.26
fig,ax=plt.subplots(figsize=(7.6,4.6))
b1=ax.bar(x-w,ind,w,label='개별 (각 AU 동역학)',color='#B0BEC5')
b2=ax.bar(x,coo,w,label='협응 (채널간 공분산)',color='#1565C0')
b3=ax.bar(x+w,both,w,label='결합',color='#2E7D32')
for bb in (b1,b2,b3):
    for r in bb:ax.text(r.get_x()+r.get_width()/2,r.get_height()+0.006,f'{r.get_height():.2f}',ha='center',fontsize=9)
ax.axhline(0.5,ls=':',color='#c0392b',lw=1);ax.text(3.4,0.51,'chance',color='#c0392b',fontsize=9)
ax.set_xticks(x);ax.set_xticklabels(corp);ax.set_ylim(0.5,0.95);ax.set_ylabel('우울 탐지 AUC')
ax.set_title('개별 동역학 vs 채널간 협응 — 협응의 보완적 기여\n(결합>개별: 4코퍼스 모두, 임상서 특히 강함)',fontsize=12)
ax.legend(fontsize=9,loc='upper right',ncol=1)
plt.tight_layout();plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig10_coord_necessity.png',dpi=145,bbox_inches='tight')
print('DONE fig10')
