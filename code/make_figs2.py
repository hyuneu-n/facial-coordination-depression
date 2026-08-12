"""
발표/논문용 그림 — 교수님(HIS/Mifu-ER) 팔레트: 파랑 #4472C4 + 주황 #ED7D31 + 강조 빨강, Blues 히트맵, Arial.
G1 데이터별 최종 AUC / G2 whole vs anchor / G3 coordination 5종 / G4 model 5종
G5 AU-AU coupling difference heatmap(데이터) / G6 face schematic(데이터기반 top pair)
모든 수치는 results CSV 검증값. 과대해석 금지 — 수치만.
"""
import numpy as np, warnings, openpyxl, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from pathlib import Path
from sklearn.covariance import ledoit_wolf
warnings.filterwarnings('ignore')
# ---- 교수님 팔레트 + 폰트 ----
for f in ['Arial','Helvetica','DejaVu Sans']:
    try: matplotlib.rcParams['font.family']=f; break
    except: pass
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':150,'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.5})
BLUE='#4472C4'; ORANGE='#ED7D31'; RED='#C00000'; GRAY='#A6A6A6'; LBLUE='#9DC3E6'
OUT='/home/hyuneun/disk_b/🟡facial-prodrome/results/'

# ========== G1 데이터별 최종 AUC ==========
fig,ax=plt.subplots(figsize=(5.2,4))
d=['CMDC\n(anchor)','DAIC-WOZ\n(anchor)','E-DAIC\n(quality gate)']; auc=[0.889,0.711,0.563]; err=[0.022,0.046,0.027]
p=[0.010,0.020,0.099]; cols=[BLUE,BLUE,GRAY]
b=ax.bar(d,auc,yerr=err,color=cols,capsize=4,width=0.6,edgecolor='k',linewidth=0.6)
ax.axhline(0.5,ls='--',c=RED,lw=1.2); ax.text(2.42,0.505,'chance',fontsize=9,color=RED,ha='right')
for i,(a,pv) in enumerate(zip(auc,p)):
    s='**' if pv<0.01 else ('*' if pv<0.05 else 'n.s.')
    ax.text(i,a+err[i]+0.02,f'{a:.3f}\n{s}',ha='center',fontweight='bold',fontsize=9)
ax.set_ylabel('AUC'); ax.set_ylim(0.4,1.0)
ax.set_title('Depression detection by anchored inter-AU coupling',fontsize=11)
plt.tight_layout(); plt.savefig(OUT+'G1_datasets.png'); plt.close()

# ========== G2 whole vs anchor (2-series grouped, HIS Fig1 style) ==========
fig,ax=plt.subplots(figsize=(5.2,4))
x=np.arange(2); w=0.34
whole=[0.873,0.579]; anchor=[0.889,0.711]           # exp31 검증값
we=[0.034,0.032]; ae=[0.022,0.046]
ax.bar(x-w/2,whole,w,yerr=we,capsize=3,label='Whole interview',color=LBLUE,edgecolor='k',lw=0.6)
ax.bar(x+w/2,anchor,w,yerr=ae,capsize=3,label='Symptom / negative anchor',color=BLUE,edgecolor='k',lw=0.6)
for i in range(2):
    ax.text(i,max(anchor[i],whole[i])+0.055,f'+{anchor[i]-whole[i]:.3f}',ha='center',color=RED,fontweight='bold',fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(['CMDC','DAIC-WOZ']); ax.axhline(0.5,ls='--',c=RED,lw=1.2)
ax.set_ylabel('AUC'); ax.set_ylim(0.4,1.0); ax.legend(fontsize=9,loc='upper right',framealpha=0.9)
ax.set_title('Effect of question anchoring (same pipeline)',fontsize=11)
plt.tight_layout(); plt.savefig(OUT+'G2_anchor.png'); plt.close()

# ========== G3 coordination measures 5종 (CMDC) ==========
fig,ax=plt.subplots(figsize=(6.2,4))
m=['Covariance\n(coupling)','Lagged corr\n(timing)','NMF\nsynergy','HMM\nstate','MdRQA\nrigidity']
v=[0.885,0.802,0.535,0.441,0.364]; e=[0.025,0.025,0.043,0.037,0.073]; cc=[BLUE,BLUE,GRAY,GRAY,GRAY]
ax.bar(m,v,yerr=e,capsize=3,color=cc,edgecolor='k',lw=0.6,width=0.65)
for i,a in enumerate(v): ax.text(i,a+e[i]+0.02,f'{a:.3f}',ha='center',fontsize=9)
ax.axhline(0.5,ls='--',c=RED,lw=1.2); ax.text(4.4,0.505,'chance',fontsize=9,color=RED,ha='right')
ax.set_ylabel('AUC'); ax.set_ylim(0.3,1.0)
ax.set_title('Coordination measures (CMDC, n=44)',fontsize=11)
plt.tight_layout(); plt.savefig(OUT+'G3_coordination.png'); plt.close()

# ========== G4 model comparison 5종 (CMDC) ==========
fig,ax=plt.subplots(figsize=(6.2,4))
m=['Tangent +\nLogReg','SPDNet','VNN','Sparse\nL1','Transformer']; v=[0.885,0.79,0.55,0.75,0.27]
cc=[BLUE,GRAY,GRAY,GRAY,GRAY]
ax.bar(m,v,color=cc,edgecolor='k',lw=0.6,width=0.65)
ax.bar(0,0.885,color=BLUE,edgecolor=RED,lw=2.0,width=0.65)  # proposed 빨강 테두리 강조(HIS식)
for i,a in enumerate(v): ax.text(i,a+0.02,f'{a:.2f}',ha='center',fontsize=9)
ax.axhline(0.5,ls='--',c=RED,lw=1.2)
ax.set_ylabel('AUC'); ax.set_ylim(0.2,1.0)
ax.set_title('Model comparison (CMDC, n=44)',fontsize=11)
plt.tight_layout(); plt.savefig(OUT+'G4_models.png'); plt.close()

# ========== 데이터 로드: CMDC anchor 공분산 per subject ==========
AUc=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45']
AUcol=[a+'_r' for a in AUc]
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUcol]
    except:return None
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if len(fe)>=10 else None
covs=[];ys=[]
for s,l in cl.items():
    segs=[cq(s,q) for q in [3,7]]
    if all(x is not None for x in segs):
        seg=np.vstack(segs);seg=(seg-seg.mean(0))/(seg.std(0)+1e-6);c,_=ledoit_wolf(seg);covs.append(c);ys.append(l)
covs=np.array(covs);ys=np.array(ys)
mdd=covs[ys==1].mean(0); hc=covs[ys==0].mean(0); diff=mdd-hc
np.fill_diagonal(diff,0)

# ========== G5 coupling difference heatmap (데이터) ==========
fig,ax=plt.subplots(figsize=(6.4,5.4))
vmax=np.abs(diff).max()
im=ax.imshow(diff,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
ax.set_xticks(range(len(AUc)));ax.set_yticks(range(len(AUc)))
ax.set_xticklabels(AUc,rotation=90,fontsize=7);ax.set_yticklabels(AUc,fontsize=7)
cb=plt.colorbar(im,fraction=0.046,pad=0.04);cb.set_label('Δ coupling  (MDD − HC)',fontsize=9)
ax.set_title('Inter-AU coupling difference (CMDC anchor)',fontsize=11)
ax.grid(False)
plt.tight_layout(); plt.savefig(OUT+'G5_coupling_heatmap.png'); plt.close()

# top |diff| pairs (데이터 산출 — 단언 금지, 데이터가 말하게)
iu=np.triu_indices(len(AUc),1)
mags=np.abs(diff[iu]); order=np.argsort(mags)[::-1][:8]
top=[(AUc[iu[0][o]],AUc[iu[1][o]],diff[iu[0][o],iu[1][o]]) for o in order]
print('TOP coupling-difference pairs (data-driven):',flush=True)
for a,b_,dv in top: print(f'  {a}-{b_}: {dv:+.3f}',flush=True)

# ========== G6 face schematic + top pairs (데이터 기반 edge) ==========
# 대략적 정면 얼굴 좌표(AU 해부학적 위치)
POS={'AU01':(-0.18,0.55),'AU02':(-0.42,0.60),'AU04':(0.0,0.50),'AU05':(-0.25,0.40),
     'AU06':(-0.55,0.15),'AU07':(-0.28,0.28),'AU09':(0.0,0.20),'AU10':(-0.15,-0.10),
     'AU12':(-0.35,-0.28),'AU14':(-0.40,-0.30),'AU15':(-0.32,-0.42),'AU17':(0.0,-0.62),
     'AU20':(-0.30,-0.35),'AU23':(0.0,-0.32),'AU25':(0.0,-0.38),'AU26':(0.0,-0.50),'AU45':(-0.30,0.42)}
# 좌우 대칭 반영: 음수 x는 왼쪽, 짝은 오른쪽에도 점. 여기선 왼쪽만 표시(간결)
fig,ax=plt.subplots(figsize=(5.6,6.4))
ax.add_patch(Ellipse((0,0),1.15,1.55,fill=False,ec='k',lw=1.3))
ax.add_patch(Ellipse((-0.28,0.40),0.30,0.16,fill=False,ec='gray',lw=1))  # 왼눈
ax.add_patch(Ellipse((0.28,0.40),0.30,0.16,fill=False,ec='gray',lw=1))   # 오른눈
ax.plot([0,0],[0.30,-0.05],c='gray',lw=1)                                # 코
ax.add_patch(Ellipse((0,-0.38),0.44,0.16,fill=False,ec='gray',lw=1))     # 입
# top pair edges (두께 ∝ |Δ|, 색 = 부호)
mx=max(abs(dv) for _,_,dv in top)
for a,b_,dv in top[:6]:
    if a in POS and b_ in POS:
        (x1,y1),(x2,y2)=POS[a],POS[b_]
        col=RED if dv>0 else BLUE
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-',lw=1+4*abs(dv)/mx,
                     color=col,alpha=0.8,connectionstyle='arc3,rad=0.15'))
for au,(x,y) in POS.items():
    ax.plot(x,y,'o',ms=6,c='k')
    ax.annotate(au,(x,y),textcoords='offset points',xytext=(4,4),fontsize=7)
ax.set_xlim(-0.75,0.75);ax.set_ylim(-0.9,0.9);ax.set_aspect('equal');ax.axis('off')
ax.set_title('Discriminative AU-coupling pairs\n(edge width ∝ |Δ coupling|; red: higher in MDD, blue: higher in HC)',fontsize=10)
plt.tight_layout(); plt.savefig(OUT+'G6_face.png'); plt.close()
print('\nALL FIGS saved: G1-G6',flush=True)
