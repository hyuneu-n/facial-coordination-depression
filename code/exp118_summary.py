"""fig15 — 통합 요약 (세미나용). 4패널: 조기탐지·협응보완성(CI)·교차코퍼스·경량. 모두 실측."""
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np, os
for c in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c):fm.fontManager.addfont(c);plt.rcParams['font.family']=fm.FontProperties(fname=c).get_name()
plt.rcParams['axes.unicode_minus']=False;plt.rcParams.update({'font.size':11})
fig,axs=plt.subplots(2,2,figsize=(12,9))
# A: 조기탐지
ax=axs[0,0];x=[20,40,60,80,100];y=[0.694,0.770,0.771,0.772,0.802]
ax.plot(x,y,'-o',color='#1565C0',lw=2.4);ax.axhline(y[-1],ls='--',color='#9E9E9E',lw=1)
ax.axvline(40,ls=':',color='#c0392b',lw=1.3);ax.text(41,0.60,'앞 40%\n≈전체 96%',color='#c0392b',fontsize=10,fontweight='bold')
for xi,yi in zip(x,y):ax.text(xi,yi+0.008,f'{yi:.2f}',ha='center',fontsize=9)
ax.set_ylim(0.5,0.87);ax.set_xlabel('관찰한 세션 앞부분 (%)');ax.set_ylabel('탐지 AUC')
ax.set_title('① 조기 탐지 — 앞부분이면 충분\n(D-Vlog, 시간모델링)',fontsize=12,fontweight='bold')
# B: 협응 보완성 Δ + CI
ax=axs[0,1]
corp=['CMDC\n(임상)','E-DAIC\n(임상)','D-Vlog\n(vlog)','LMVD\n(vlog)'];dl=[0.103,0.029,0.007,0.002]
lo=[0.077,0.009,0.003,-0.001];hi=[0.132,0.050,0.011,0.004]
err=[[d-l for d,l in zip(dl,lo)],[h-d for d,h in zip(dl,hi)]]
cols=['#1565C0' if l>0 else '#B0BEC5' for l in lo]
ax.bar(range(4),dl,yerr=err,color=cols,capsize=4)
ax.axhline(0,color='#333',lw=0.8)
for i,d in enumerate(dl):ax.text(i,hi[i]+0.004,f'+{d:.3f}',ha='center',fontsize=9,fontweight='bold')
ax.set_xticks(range(4));ax.set_xticklabels(corp);ax.set_ylabel('결합 - 개별 AUC (Δ)')
ax.set_title('② 협응이 보완 정보 추가 (robust)\n결합>개별, 임상서 강함 (95% CI)',fontsize=12,fontweight='bold')
# C: 교차 코퍼스 (정직)
ax=axs[1,0];cc=['CMDC','D-Vlog','LMVD','E-DAIC'];au=[0.89,0.81,0.63,0.62]
col2=['#2E7D32','#2E7D32','#9E9E9E','#9E9E9E']
b=ax.bar(cc,au,color=col2);ax.axhline(0.5,ls='--',color='#c0392b',lw=1)
for r,v in zip(b,au):ax.text(r.get_x()+r.get_width()/2,v+0.01,f'{v:.2f}',ha='center',fontweight='bold',fontsize=10)
ax.set_ylim(0.5,1.0);ax.set_ylabel('AUC')
ax.text(0.5,0.94,'강',color='#2E7D32',fontsize=11,ha='center',transform=ax.get_xaxis_transform())
ax.text(2.5,0.70,'약',color='#9E9E9E',fontsize=11,ha='center',transform=ax.get_xaxis_transform())
ax.set_title('③ 맥락 의존 (정직한 범위)\n표현적·고품질서 강, 임상인터뷰서 약',fontsize=12,fontweight='bold')
# D: 경량
ax=axs[1,1];mods=['제안\n(SPD+GRU)','Transformer','딥 멀티모달\nSOTA'];par=[67745,267777,5000000];labs=['68K','268K','~5M']
order=[2,1,0];colr=['#1565C0','#90A4AE','#B0BEC5']
b=ax.barh([mods[i] for i in order],[par[i] for i in order],color=[colr[i] for i in order]);ax.set_xscale('log')
for k,i in enumerate(order):ax.text(par[i]*1.2,k,labs[i],va='center',fontweight='bold',fontsize=10)
ax.set_xlim(3e4,2e7);ax.set_xlabel('파라미터 (log)')
ax.set_title('④ 온디바이스 경량 — 68K · CPU 2ms\n(SOTA 대비 ~100배 작음)',fontsize=12,fontweight='bold')
fig.suptitle('표정 coordination 기반 우울 인식 — 핵심 결과 요약',fontsize=15,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig15_summary.png',dpi=145,bbox_inches='tight')
print('DONE fig15_summary.png')
