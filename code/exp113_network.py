"""exp113 — 협응 네트워크 위상. 협응 행렬을 그래프로: 밀도·클러스터링·차수이질성·참여율(PR=복잡도)·스펙트럴갭.
우울군 네트워크 구조 다른가? 4코퍼스 재현. 네트워크 속성은 pairwise보다 robust할 수도.
"""
import numpy as np, warnings, os, re, glob, csv
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/'
def netfeats(series):
    Z=(series-series.mean(0))/(series.std(0)+1e-6);C=np.corrcoef(Z.T)
    if np.isnan(C).any():return None
    d=len(C);A=np.abs(C.copy());np.fill_diagonal(A,0)
    off=A[np.triu_indices(d,1)]
    density=off.mean();het=A.sum(1).std()  # 밀도, 차수 이질성
    # 클러스터링(임계 이진화 transitivity)
    thr=np.median(off);B=(A>thr).astype(float);np.fill_diagonal(B,0)
    B2=B@B;tri=np.trace(B@B2);paths=B2.sum()-np.trace(B2)
    clust=tri/paths if paths>0 else 0
    # 참여율 PR = 유효 협응모드수(복잡도), 스펙트럴갭
    ev=np.sort(np.linalg.eigvalsh(C))[::-1];ev=np.clip(ev,0,None)
    PR=(ev.sum()**2)/((ev**2).sum()+1e-9)
    gap=ev[0]-ev[1]
    return [density,het,clust,PR,gap]
NAMES=['density','차수이질성','clustering','PR(복잡도)','스펙트럴갭']
def persec(f):
    try:df=pd.read_csv(f,usecols=lambda c:c.strip() in set(['timestamp','success','confidence']+AUS))
    except:return None
    df.columns=[c.strip() for c in df.columns]
    if 'timestamp' not in df or not set(AUS).issubset(df.columns):return None
    if 'success' in df:df=df[df['success']==1]
    if len(df)<50:return None
    df=df.copy();df['sec']=np.floor(df['timestamp']).astype(int)
    return df.groupby('sec')[AUS].mean().values
allres={}
def report(name,F,Y):
    F=np.array(F);Y=np.array(Y);print(f'\n#### {name} n={len(Y)} ####',flush=True);d={}
    for k,nm in enumerate(NAMES):
        a=F[Y==1,k];b=F[Y==0,k]
        try:_,p=mannwhitneyu(a,b,alternative='two-sided')
        except:p=1
        try:auc=roc_auc_score(Y,F[:,k]);auc=max(auc,1-auc)
        except:auc=.5
        dr='MDD>HC' if a.mean()>b.mean() else 'MDD<HC'
        print(f'  {nm:12s} MDD={a.mean():.3f} HC={b.mean():.3f} [{dr}] p={p:.2e} AUC={auc:.3f}',flush=True)
        d[nm]=(dr,p,auc)
    allres[name]=d
# D-Vlog
R=Path(DATA+'D-Vlog')
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
RAW=[];Y=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy'
    if not fv.exists():continue
    try:V=np.load(fv)
    except:continue
    if V.ndim!=2 or V.shape[0]<50 or V.shape[1]!=136:continue
    RAW.append(nf(V));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
pca=PCA(17).fit(np.vstack([RAW[i][::5] for i in range(len(RAW))]))
F=[];Yv=[]
for x,y in zip(RAW,Y):
    f=netfeats(pca.transform(x))
    if f:F.append(f);Yv.append(y)
report('D-Vlog',F,Yv)
# CMDC
F=[];Y=[]
for dfold in sorted(glob.glob(DATA+'CMDC/extracted/*/')):
    name=os.path.basename(dfold.rstrip('/'))
    if not(name.startswith('HC') or name.startswith('MDD')):continue
    segs=[persec(f2) for f2 in glob.glob(dfold+'Q*.csv') if re.match(r'^Q\d+\.csv$',os.path.basename(f2))]
    segs=[s for s in segs if s is not None]
    if not segs:continue
    s=np.vstack(segs)
    if len(s)<40:continue
    f=netfeats(s)
    if f:F.append(f);Y.append(1 if name.startswith('MDD') else 0)
report('CMDC',F,Y)
# E-DAIC
lab={}
for spn in ['train','dev','test']:
    p=DATA+f'E-DAIC/labels/{spn}_split.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            pid=r.get('Participant_ID','').strip();b=r.get('PHQ_Binary','').strip()
            if pid and b in('0','1'):lab[pid]=int(b)
F=[];Y=[]
for dfold in sorted(glob.glob(DATA+'E-DAIC/extracted/*_P/')):
    pid=os.path.basename(dfold.rstrip('/')).replace('_P','')
    if pid not in lab:continue
    fp=dfold+f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'
    if not os.path.exists(fp):continue
    ps=persec(fp)
    if ps is None or len(ps)<40:continue
    f=netfeats(ps)
    if f:F.append(f);Y.append(lab[pid])
report('E-DAIC',F,Y)
# LMVD
def lm(i):return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
F=[];Y=[]
for f2 in sorted(glob.glob(DATA+'LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(f2))
    if not m:continue
    ps=persec(f2)
    if ps is None or len(ps)<40:continue
    f=netfeats(ps)
    if f:F.append(f);Y.append(lm(int(m.group(1))))
report('LMVD',F,Y)
# 재현 요약
print('\n===== feature별 4코퍼스 방향 일치 =====',flush=True)
for nm in NAMES:
    line=[]
    for c in ['D-Vlog','CMDC','E-DAIC','LMVD']:
        if c in allres and nm in allres[c]:
            dr,p,a=allres[c][nm];line.append(f'{c}:{dr}({"*" if p<0.05 else "ns"})')
    dirs=[x.split(':')[1].split('(')[0] for x in line]
    agree='★일치' if len(set(dirs))==1 else ('3/4' if max([dirs.count(x) for x in set(dirs)])>=3 else '')
    print(f'  {nm:12s} '+' | '.join(line)+f'  {agree}',flush=True)
print('DONE',flush=True)
