"""개념 설명용 시각자료 3종 (교수님 팔레트, 영어 라벨).
C1 anchor 타임라인 / C2 coupling(공분산) / C3 Riemannian tangent."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
BLUE='#4472C4'; NAVY='#002852'; LBLUE='#9DC3E6'; RED='#C00000'; GRAY='#B7BccC'.replace('c','B'); GRAY='#B0B4BB'
OUT='/home/hyuneun/disk_b/🟡facial-prodrome/results/'

# ============ C1 anchor timeline ============
fig,ax=plt.subplots(figsize=(11,3.2))
T=10.0
# AU signal (jittery) on top
t=np.linspace(0,T,1000); rng=np.random.RandomState(1)
sig=0.5+0.15*np.sin(t*3)+0.08*rng.randn(1000); sig=np.convolve(sig,np.ones(15)/15,'same')
ax.plot(t,sig*0.6+1.5,color=NAVY,lw=1.0)
ax.text(-0.15,1.8,'AU\nsignal',ha='right',va='center',fontsize=11,color=NAVY)
# interview bar
ax.add_patch(Rectangle((0,0.4),T,0.5,facecolor=GRAY,edgecolor='k',lw=0.8))
# question segments
segs=[(0.5,1.4,'general\ntalk',False),(2.2,3.3,'sleep?\n(symptom)',True),
      (4.0,4.8,'hobby?\n(general)',False),(5.6,6.9,'feel down?\n(negative)',True),
      (8.0,9.1,'fatigue?\n(symptom)',True)]
for a,b,lab,anc in segs:
    c=BLUE if anc else GRAY
    ax.add_patch(Rectangle((a,0.4),b-a,0.5,facecolor=c,edgecolor='k',lw=0.8))
    ax.text((a+b)/2,0.15,lab,ha='center',va='top',fontsize=9,color=(BLUE if anc else '#666'))
    if anc:  # shade the AU signal window
        ax.axvspan(a,b,ymin=0.55,ymax=0.95,color=BLUE,alpha=0.12)
        ax.annotate('',xy=((a+b)/2,1.15),xytext=((a+b)/2,0.95),
                    arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.4))
ax.text(T/2,1.02,'↑ analyze facial coupling ONLY in these windows',ha='center',fontsize=10,color=BLUE,fontweight='bold')
ax.set_xlim(-1.2,T+0.3); ax.set_ylim(-0.4,2.1); ax.axis('off')
ax.set_title('Question anchoring — select symptom / negative question windows',fontsize=13,color=NAVY,fontweight='bold')
# legend
ax.add_patch(Rectangle((0.2,-0.35),0.4,0.18,facecolor=BLUE)); ax.text(0.75,-0.26,'anchor (symptom/negative)',fontsize=9,va='center')
ax.add_patch(Rectangle((4.6,-0.35),0.4,0.18,facecolor=GRAY)); ax.text(5.15,-0.26,'discarded (general talk)',fontsize=9,va='center')
plt.tight_layout(); plt.savefig(OUT+'C1_anchor.png'); plt.close()

# ============ C2 coupling ============
fig,axes=plt.subplots(1,3,figsize=(12,3.4),gridspec_kw={'width_ratios':[1,1,0.8]})
t=np.linspace(0,6,400); rng=np.random.RandomState(3)
base=np.sin(t*2)
# HC: two AUs move together
axes[0].plot(t,base+0.1*rng.randn(400)+2,color=BLUE,lw=1.6,label='AU12 (smile)')
axes[0].plot(t,base+0.1*rng.randn(400),color=NAVY,lw=1.6,label='AU25 (lips)')
axes[0].set_title('High coupling\n(muscles move together)',fontsize=12,color=NAVY)
axes[0].legend(fontsize=9,loc='upper right'); axes[0].set_yticks([]); axes[0].set_xlabel('time')
# MDD: independent
axes[1].plot(t,np.sin(t*2)+0.1*rng.randn(400)+2,color=BLUE,lw=1.6)
axes[1].plot(t,np.sin(t*3.7+1)+0.1*rng.randn(400),color=NAVY,lw=1.6)
axes[1].set_title('Low coupling\n(muscles move independently)',fontsize=12,color=NAVY)
axes[1].set_yticks([]); axes[1].set_xlabel('time')
# covariance matrix mini
M=np.array([[1,.8,.2],[.8,1,.15],[.2,.15,1]])
im=axes[2].imshow(M,cmap='RdBu_r',vmin=-1,vmax=1)
axes[2].set_xticks([0,1,2]);axes[2].set_yticks([0,1,2])
axes[2].set_xticklabels(['AU12','AU25','AU06'],fontsize=9);axes[2].set_yticklabels(['AU12','AU25','AU06'],fontsize=9)
for i in range(3):
    for j in range(3): axes[2].text(j,i,f'{M[i,j]:.1f}',ha='center',va='center',fontsize=9,color='k')
axes[2].set_title('covariance matrix\n= all pairwise couplings',fontsize=12,color=NAVY)
fig.suptitle('Coupling = how much two facial muscles co-activate (measured by covariance)',fontsize=13,color=NAVY,fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.90]); plt.savefig(OUT+'C2_coupling.png'); plt.close()

# ============ C3 Riemannian tangent ============
fig,ax=plt.subplots(figsize=(9,4.6))
# curved manifold (arc band)
xs=np.linspace(0,10,200); curve=2.2+1.4*np.sin(xs*0.32)
ax.plot(xs,curve,color=NAVY,lw=2.2)
ax.fill_between(xs,curve-0.12,curve+0.12,color=LBLUE,alpha=0.5)
ax.text(9.6,curve[-1]+0.35,'SPD manifold\n(curved space of\ncovariance matrices)',fontsize=10,color=NAVY,ha='right')
# points on manifold
px=np.array([1.5,3.0,4.6,6.2,7.6]); py=2.2+1.4*np.sin(px*0.32)
ax.scatter(px,py,s=70,color=BLUE,edgecolor='k',zorder=5)
for x,y in zip(px,py): ax.text(x,y+0.22,'Σ',fontsize=11,color=BLUE,ha='center')
# mean
mx=4.6; my=2.2+1.4*np.sin(mx*0.32)
ax.scatter([mx],[my],s=140,color=RED,edgecolor='k',zorder=6,marker='*')
ax.text(mx,my-0.5,'Riemannian mean  Σ̄',fontsize=10,color=RED,ha='center')
# tangent line at mean
slope=1.4*0.32*np.cos(mx*0.32)
tx=np.linspace(mx-3.2,mx+3.4,50); ty=my+slope*(tx-mx)
ax.plot(tx,ty,color='#444',lw=1.8,ls='-')
ax.text(tx[-1],ty[-1]+0.15,'tangent space (flat)\n→ ordinary vectors, linear classifier OK',fontsize=10,color='#333',ha='right')
# projection (log map) of points onto tangent line
for x,y in zip(px,py):
    # foot on tangent line
    t0=((x-mx)+ (y-my)*slope)/(1+slope**2); fx=mx+t0; fy=my+slope*t0
    ax.plot([x,fx],[y,fy],ls=':',color=BLUE,lw=1.1)
    ax.scatter([fx],[fy],s=45,color=BLUE,marker='s',zorder=5)
ax.annotate('log map',xy=(px[1]+0.1,(py[1]+ (my+slope*(((px[1]-mx)+(py[1]-my)*slope)/(1+slope**2))))/2),
            fontsize=10,color=BLUE)
# euclidean wrong line between two far points
ax.plot([px[0],px[-1]],[py[0],py[-1]],ls='--',color=GRAY,lw=1.6)
ax.text((px[0]+px[-1])/2,(py[0]+py[-1])/2-0.55,'straight (Euclidean) distance = WRONG here',fontsize=9.5,color=GRAY,ha='center')
ax.set_xlim(-0.3,10.5); ax.set_ylim(0.2,4.4); ax.axis('off')
ax.set_title('Why Riemannian? Covariances live on a curved space — flatten via tangent map first',fontsize=12.5,color=NAVY,fontweight='bold')
plt.tight_layout(); plt.savefig(OUT+'C3_riemannian.png'); plt.close()
print('DONE C1,C2,C3',flush=True)
