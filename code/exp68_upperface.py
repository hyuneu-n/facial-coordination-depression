"""
exp68 — E-DAIC 규명 + 강화: 상부얼굴(발화-강건) expressivity가 임상 인터뷰서도 되나?
가설: 임상 인터뷰=말하기→하부얼굴(입 AU25/26/12) 발화 오염→expressivity 신호 죽음.
 상부얼굴(눈썹·눈 AU1/2/4/5/6/7/9)=발화 덜 오염+감정특이→E-DAIC/DAIC서도 hypoexpressivity 생존?
검증: 코퍼스별 tension(평균AU) MDD vs HC 효과크기 — 상부 vs 하부 vs 전체.
 + coupling+expressivity(상부) AUC가 E-DAIC/DAIC서 개선되나. 결과: results/exp68_upperface.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10
UPPER=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r']  # 상부(발화 강건)
LOWER=['AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']  # 하부(발화 오염)
def cliffs(a,b):
    a=np.asarray(a);b=np.asarray(b);return ((a[:,None]>b[None,:]).sum()-(a[:,None]<b[None,:]).sum())/(len(a)*len(b))
def cov_of(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def cv(COV,F,y,mode):
    COV=np.array(COV);F=np.nan_to_num(np.array(F));y=np.array(y);a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            if mode=='exp':X=F
            else:
                T=TangentSpace(metric='riemann').fit(COV[tr]).transform(COV);X=T if mode=='cov' else np.column_stack([T,F])
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a)
def analyze(subjAU, y, colidx, name):
    """subjAU: per-subj full AU array; colidx: dict AU_r->col index in that array"""
    y=np.array(y)
    up=[c for c in UPPER if c in colidx]; lo=[c for c in LOWER if c in colidx]
    ui=[colidx[c] for c in up]; li=[colidx[c] for c in lo]; alli=list(colidx.values())
    # tension 벡터(pop 표준화)
    allmean=np.array([au.mean(0) for au in subjAU])
    pm=allmean.mean(0);ps=allmean.std(0)+1e-6; T=(allmean-pm)/ps
    def eff(cols):
        # 평균 활성 전반(해당 부위 tension 합)
        v=T[:,cols].mean(1); d=cliffs(v[y==1],v[y==0]);_,p=mannwhitneyu(v[y==1],v[y==0]);return d,p
    du,pu=eff([list(colidx.values()).index(i) for i in ui])
    dl,pl=eff([list(colidx.values()).index(i) for i in li])
    da,pa=eff(list(range(len(alli))))
    print(f'\n===== {name} (n={len(y)}, 우울{y.sum()}) tension 효과크기(cliffδ, MDD-HC) =====',flush=True)
    print(f'  상부얼굴 δ={du:+.3f} p={pu:.4f} | 하부얼굴 δ={dl:+.3f} p={pl:.4f} | 전체 δ={da:+.3f} p={pa:.4f}',flush=True)
    # AUC: coupling(all) vs +expressivity(upper) vs +expressivity(all)
    COV=[cov_of(au) for au in subjAU]
    EXPu=T[:,[list(colidx.values()).index(i) for i in ui]]
    EXPa=T
    rc=cv(COV,EXPu,y,'cov'); rbu=cv(COV,EXPu,y,'comb'); rba=cv(COV,EXPa,y,'comb')
    print(f'  coupling={rc:.3f} | +상부expr={rbu:.3f}(Δ{rbu-rc:+.3f}) | +전체expr={rba:.3f}(Δ{rba-rc:+.3f})',flush=True)
    return [name,len(y),du,pu,dl,pl,rc,rbu,rba]
out=[]
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
# CMDC
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
A=[];Y=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: A.append(np.vstack(parts));Y.append(l)
out.append(analyze(A,Y,{c:i for i,c in enumerate(AUc)},'CMDC'))
# DAIC
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
def dau(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None
    h=[x.strip() for x in open(p).readline().split(',')];oi=h.index('success');ai=[h.index(c) for c in AUd]
    fe=[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if fe else None
A=[];Y=[]
for pid,l in dl.items():
    au=dau(pid)
    if au is not None and len(au)>=60: A.append(au);Y.append(l)
out.append(analyze(A,Y,{c:i for i,c in enumerate(AUd)},'DAIC'))
# E-DAIC
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
def eau(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*.csv'))
    if not fs:return None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[]
    for ln in open(fs[0]).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if len(fe)>=60 else None
A=[];Y=[]
for pid,l in el.items():
    au=eau(pid)
    if au is not None: A.append(au);Y.append(l)
out.append(analyze(A,Y,{c:i for i,c in enumerate(AUc)},'E-DAIC'))
# LMVD
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423): return 1
    if (602<=i<=1116) or (1425<=i<=1824): return 0
    return None
A=[];Y=[]
for f in sorted(glob.glob(str(V/'*.csv'))):
    l=lab(int(Path(f).stem))
    if l is None:continue
    h=[x.strip() for x in open(f).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:continue
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    if len(fe)>=60: A.append(np.nan_to_num(np.array(fe)));Y.append(l)
out.append(analyze(A,Y,{c:i for i,c in enumerate(AUc)},'LMVD'))
print('\n판정: 상부얼굴 tension δ가 임상(DAIC/E-DAIC)서 전체보다 크고 유의 & +상부expr가 개선 →',flush=True)
print('      "발화-강건 상부얼굴 hypoexpressivity가 임상+야생 모두 일반화" = E-DAIC 복구+강화.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp68_upperface.csv','w') as f:
    f.write('corpus,n,upper_delta,upper_p,lower_delta,lower_p,coupling,comb_upper,comb_all\n')
    for r in [x for x in out if x]:f.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp68_upperface.csv',flush=True)
