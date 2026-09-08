"""exp99 — 전조 시계열 figure. 세션 내 coordination이 시간에 따라 어떻게 변하나.
Panel A: 예시 피험자 coordination 궤적(2D, 시간 색). Panel B: 그룹 시계열(정규화 세션시간별 coordination 변화, MDD vs HC).
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np, csv, os, warnings
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
KOR=None
for c in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c):
        fm.fontManager.addfont(c);KOR=fm.FontProperties(fname=c).get_name();plt.rcParams['font.family']=KOR
plt.rcParams['axes.unicode_minus']=False;plt.rcParams.update({'font.size':12})
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
W=20; STR=10; NT=20  # 윈도우 20s, 정규화 시간 20점
def nf(V):
    T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
def logm_vec(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None);L=U@np.diag(np.log(ev))@U.T
    iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def win_seq(seq,d=20):
    outs=[]
    for s in range(0,len(seq)-W+1,STR):
        seg=seq[s:s+W]
        if seg.std(0).min()<1e-8:seg=seg+np.random.randn(*seg.shape)*1e-6
        C,_=ledoit_wolf(seg);outs.append(logm_vec(C+1e-3*np.eye(d)))
    return np.array(outs)
raw=[];Y=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:V=np.load(fv)
    except:continue
    if V.ndim!=2 or V.shape[0]<W+4*STR or V.shape[1]!=136:continue
    raw.append(nf(V));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
Y=np.array(Y)
pca=PCA(20).fit(np.vstack([raw[i][::5] for i in range(len(raw))]))
# 각 피험자 tangent 시퀀스
seqs=[];velN=[]
for i in range(len(Y)):
    Z=pca.transform(raw[i]);Z=(Z-Z.mean(0))/(Z.std(0)+1e-6)
    s=win_seq(Z)
    if len(s)<5:seqs.append(None);velN.append(None);continue
    seqs.append(s)
    v=np.linalg.norm(np.diff(s,axis=0),axis=1)  # 윈도우별 coordination 변화량
    idx=np.linspace(0,len(v)-1,NT);velN.append(np.interp(idx,np.arange(len(v)),v))
# 2D PCA (모든 tangent) for Panel A
allS=np.vstack([s for s in seqs if s is not None])
p2=PCA(2).fit(allS)
fig,axs=plt.subplots(1,2,figsize=(13.5,5.4))
# --- Panel A: 예시 궤적 (로버스트 축 + 1 HC + 1 MDD) ---
ax=axs[0]
allP2=p2.transform(allS)
xlo,xhi=np.percentile(allP2[:,0],[3,97]);ylo,yhi=np.percentile(allP2[:,1],[3,97])
def pick(label):
    # 축 안에 잘 들어오고 길이 충분한 대표 피험자 선택
    for i in range(len(Y)):
        if Y[i]!=label or seqs[i] is None or len(seqs[i])<10:continue
        P2=p2.transform(seqs[i])
        if (P2[:,0]>=xlo).all() and (P2[:,0]<=xhi).all() and (P2[:,1]>=ylo).all() and (P2[:,1]<=yhi).all():
            return P2
    return None
sc=None;allx=[];ally=[]
for label,color,nm in [(0,'#1565C0','건강군'),(1,'#c0392b','우울군')]:
    P2=pick(label)
    if P2 is None:continue
    allx+=list(P2[:,0]);ally+=list(P2[:,1])
    ax.plot(P2[:,0],P2[:,1],'-',color=color,alpha=0.45,lw=1.5,zorder=2)
    sc=ax.scatter(P2[:,0],P2[:,1],c=np.linspace(0,100,len(P2)),cmap='viridis',s=40,zorder=3,edgecolor=color,linewidth=0.6)
    ax.scatter(P2[0,0],P2[0,1],marker='o',s=130,edgecolor=color,facecolor='white',zorder=4,lw=2.2)
    ax.scatter(P2[-1,0],P2[-1,1],marker='s',s=120,color=color,zorder=4)
    ax.annotate(nm,(P2[-1,0],P2[-1,1]),fontsize=11,color=color,fontweight='bold',xytext=(6,6),textcoords='offset points')
if allx:
    mx=(max(allx)-min(allx))*0.15+0.3;my=(max(ally)-min(ally))*0.15+0.3
    ax.set_xlim(min(allx)-mx,max(allx)+mx);ax.set_ylim(min(ally)-my,max(ally)+my)
if sc is not None:
    cb=plt.colorbar(sc,ax=ax,fraction=0.045,pad=0.02);cb.set_label('세션 진행 (%)',fontsize=10)
ax.set_title('세션 내 coordination 궤적 (예시 2명)\n○시작 → ■끝, 점 색=시간 진행',fontsize=12)
ax.set_xlabel('coordination 축 1');ax.set_ylabel('coordination 축 2')
# --- Panel B: 그룹 시계열 ---
ax=axs[1]
VM=np.array([velN[i] for i in range(len(Y)) if velN[i] is not None and Y[i]==1])
VH=np.array([velN[i] for i in range(len(Y)) if velN[i] is not None and Y[i]==0])
t=np.linspace(0,100,NT)
for V,c,lab in [(VH,'#1565C0','건강군 (HC)'),(VM,'#c0392b','우울군 (MDD)')]:
    m=V.mean(0);se=V.std(0)/np.sqrt(len(V))
    ax.plot(t,m,'-',color=c,lw=2.2,label=lab);ax.fill_between(t,m-se,m+se,color=c,alpha=0.18)
ax.set_title('세션 내 coordination 변화량 시계열\n(정규화 세션시간별, 그룹평균±SE)',fontsize=12)
ax.set_xlabel('세션 진행 (%)');ax.set_ylabel('coordination 변화량 (window간)')
ax.legend(fontsize=10,loc='best')
fig.suptitle('Within-session coordination 시간축 분석 — D-Vlog (전조 분석의 토대)',fontsize=14)
plt.tight_layout();plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig6_precursor_ts.png',dpi=145,bbox_inches='tight');plt.close()
print(f'MDD n={len(VM)} HC n={len(VH)}')
print('전체 평균 변화량 MDD=%.3f HC=%.3f'%(VM.mean(),VH.mean()))
print('DONE fig6_precursor_ts.png')
