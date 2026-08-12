"""
make_figs3 — F2 나머지 그림 (교수님 팔레트). 전부 검증된 수치.
G4 모델비교(CMDC+DAIC, 수정본) / G3b coordination 3데이터 / G7 회귀 / G8 파이프라인 / G9 격자얼굴
"""
import numpy as np, warnings, openpyxl, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
BLUE='#4472C4'; LBLUE='#9DC3E6'; ORANGE='#ED7D31'; RED='#C00000'; GRAY='#A6A6A6'; GREEN='#548235'
OUT='/home/hyuneun/disk_b/🟡facial-prodrome/results/'

# ===== G4 모델비교 (수정: Transformer 0.676) 2패널 =====
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4),sharey=True)
cm=[('Tangent+LR',0.885),('SPDNet',0.790),('Sparse-L1',0.754),('Transformer',0.676),('VNN',0.554)]
dm=[('Tangent+LR',0.711),('Sparse-L1',0.759),('SPDNet',0.661),('VNN',0.502)]
for ax,data,ttl in [(a1,cm,'(a) CMDC (n=44)'),(a2,dm,'(b) DAIC-WOZ (n=94)')]:
    names=[d[0] for d in data]; vals=[d[1] for d in data]
    cols=[BLUE if 'Tangent' in n or (n=='Sparse-L1' and ttl.startswith('(b)')) else GRAY for n in names]
    bars=ax.bar(names,vals,color=cols,edgecolor='k',lw=0.6,width=0.65)
    # 제안(Tangent) 빨강 테두리
    for i,n in enumerate(names):
        if 'Tangent' in n: bars[i].set_edgecolor(RED); bars[i].set_linewidth(2.2)
    for i,v in enumerate(vals): ax.text(i,v+0.015,f'{v:.2f}',ha='center',fontsize=9)
    ax.axhline(0.5,ls='--',c=RED,lw=1.1); ax.set_ylim(0.4,1.0); ax.set_title(ttl,fontsize=11)
    ax.tick_params(axis='x',rotation=30)
a1.set_ylabel('AUC')
fig.suptitle('Model comparison — lightweight linear beats heavy deep models',fontsize=12)
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(OUT+'G4_models.png'); plt.close()

# ===== G3b coordination 3데이터 =====
meas=['Covariance','Lagged\ncorr','NMF','HMM','MdRQA']
cmdc=[0.885,0.802,0.535,0.441,0.364]
daic=[0.711,0.723,0.741,0.549,0.580]
edaic=[0.563,0.511,0.561,0.486,np.nan]
fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,vals,ttl in [(axes[0],cmdc,'(a) CMDC'),(axes[1],daic,'(b) DAIC-WOZ'),(axes[2],edaic,'(c) E-DAIC (no anchor)')]:
    cols=[BLUE if (not np.isnan(v) and v>=0.65) else GRAY for v in vals]
    vplot=[0 if np.isnan(v) else v for v in vals]
    ax.bar(meas,vplot,color=cols,edgecolor='k',lw=0.6,width=0.7)
    for i,v in enumerate(vals):
        ax.text(i,(0.02 if np.isnan(v) else v+0.015),('n/a' if np.isnan(v) else f'{v:.2f}'),ha='center',fontsize=8)
    ax.axhline(0.5,ls='--',c=RED,lw=1.1); ax.set_ylim(0.3,1.0); ax.set_title(ttl,fontsize=11)
    ax.tick_params(axis='x',labelsize=8)
axes[0].set_ylabel('AUC')
fig.suptitle('Coordination measures across datasets (state/recurrence fail; E-DAIC: no anchor → all ~chance)',fontsize=11)
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(OUT+'G3_coordination.png'); plt.close()

# ===== G7 회귀 (CCC + MAE) =====
fig,(a1,a2)=plt.subplots(1,2,figsize=(9,4))
ds=['CMDC','DAIC-WOZ','E-DAIC']; ccc=[0.377,0.094,0.200]; cccp=[0.020,0.129,0.010]; mae=[5.84,4.81,5.01]
cols=[BLUE if p<0.05 else GRAY for p in cccp]
a1.bar(ds,ccc,color=cols,edgecolor='k',lw=0.6,width=0.6)
for i,(v,p) in enumerate(zip(ccc,cccp)):
    s='*' if p<0.05 else 'n.s.'; a1.text(i,v+0.008,f'{v:.3f}\n{s}',ha='center',fontsize=9,fontweight='bold')
a1.axhline(0,c='k',lw=0.8); a1.set_ylabel('CCC (severity regression)'); a1.set_ylim(0,0.5); a1.set_title('(a) CCC — higher better',fontsize=11)
a2.bar(ds,mae,color=ORANGE,edgecolor='k',lw=0.6,width=0.6)
for i,v in enumerate(mae): a2.text(i,v+0.05,f'{v:.2f}',ha='center',fontsize=9)
a2.set_ylabel('MAE (PHQ points)'); a2.set_ylim(0,7); a2.set_title('(b) MAE — lower better',fontsize=11)
fig.suptitle('Severity regression (detection is primary; severity is secondary)',fontsize=12)
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(OUT+'G7_regression.png'); plt.close()

# ===== G8 Method 파이프라인 다이어그램 =====
fig,ax=plt.subplots(figsize=(12,2.6)); ax.set_xlim(0,12); ax.set_ylim(0,2); ax.axis('off')
steps=[('Interview\nvideo',LBLUE),('OpenFace\nAU extraction',LBLUE),('Question anchor\n(symptom/neg)',BLUE),
       ('AU-pair\ncovariance',BLUE),('Riemannian\ntangent map',BLUE),('Logistic\nclassifier',GREEN),('Depressed?\n/ severity',ORANGE)]
w=1.5; gap=0.2; x=0.15
for i,(t,c) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((x,0.55),w,0.9,boxstyle='round,pad=0.03',fc=c,ec='k',lw=0.8,alpha=0.9))
    ax.text(x+w/2,1.0,t,ha='center',va='center',fontsize=9,color='white' if c in(BLUE,GREEN,ORANGE) else 'black',fontweight='bold')
    if i<len(steps)-1:
        ax.add_patch(FancyArrowPatch((x+w,1.0),(x+w+gap,1.0),arrowstyle='-|>',mutation_scale=14,color='k',lw=1.2))
    x+=w+gap
ax.text(6,0.2,'core contribution: anchor + inter-AU coupling on the Riemannian manifold',ha='center',fontsize=9,style='italic',color=RED)
plt.tight_layout(); plt.savefig(OUT+'G8_pipeline.png'); plt.close()

# ===== G9 격자 얼굴 (AU 설명용, 좌표 격자 + 부위 라벨) =====
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0]); iID=hd.index('ID')
subs=[str(r[iID]).strip() for r in rows[1:] if r[iID] is not None]
def lm_mean(subj,q=3):
    f=C/subj/f'Q{q}.csv'
    if not f.exists(): return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try: xi=[h.index(f'x_{i}') for i in range(68)]; yi=[h.index(f'y_{i}') for i in range(68)]; oi=h.index('success')
    except: return None
    xs=[];ys=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1: continue
            xs.append([float(v[i]) for i in xi]); ys.append([float(v[i]) for i in yi])
        except: pass
    if len(xs)<5: return None
    return np.array(xs).mean(0),np.array(ys).mean(0)
allx=[];ally=[]
for s in subs:
    r=lm_mean(s)
    if r is not None: allx.append(r[0]); ally.append(r[1])
mx=np.mean(allx,0); my=-np.mean(ally,0)  # y flip
# 정규화(눈사이 거리)
le=np.array([mx[36:42].mean(),my[36:42].mean()]); re=np.array([mx[42:48].mean(),my[42:48].mean()])
d=np.linalg.norm(le-re)+1e-6; cx,cy=mx[30],my[30]
mx=(mx-cx)/d; my=(my-cy)/d
PARTS={'jaw':range(0,17),'r_brow':range(17,22),'l_brow':range(22,27),'nose':range(27,36),'r_eye':range(36,42),'l_eye':range(42,48),'mouth_out':range(48,60),'mouth_in':range(60,68)}
fig,ax=plt.subplots(figsize=(5.5,6))
for nm,ii in PARTS.items():
    ii=list(ii); ax.plot(mx[ii],my[ii],c=BLUE,lw=1.6)
ax.scatter(mx,my,c=RED,s=14,zorder=3)
# AU 위치 라벨
labels={'AU1/2/4 (brows)':(mx[17:27].mean(),my[17:27].mean()+0.15),'AU45 (blink)':(mx[42:48].mean(),my[42:48].mean()+0.12),
        'AU6 (cheek)':(mx[1],my[1]),'AU12/15 (lip corner)':(mx[48],my[48]-0.15),'AU25/26 (lips/jaw)':(mx[57],my[57]-0.18)}
for t,(x,y) in labels.items(): ax.annotate(t,(x,y),fontsize=8,color='k',ha='center')
ax.grid(True,ls=':',alpha=0.5); ax.set_aspect('equal')
ax.set_title('Facial landmarks & AU locations (coordinate grid)\nmean face, CMDC Q3',fontsize=11)
plt.tight_layout(); plt.savefig(OUT+'G9_grid_face.png'); plt.close()
print('DONE → G4,G3,G7,G8,G9',flush=True)
