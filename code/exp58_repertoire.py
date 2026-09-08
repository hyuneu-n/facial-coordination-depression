"""
exp58 — idea B: 표정 "지도" 위상 = expression repertoire 협착 (VPR 이식).
각 사람의 표정상태를 manifold 위 '장소'로 보고 repertoire 특성 측정.
 특징: PR(participation ratio=유효 표정차원, amplitude-invariant, 워크플로 미검증 아이디어),
       occ_entropy(표정상태 occupancy 엔트로피=repertoire 풍부도),
       trans_entropy(상태 전이 예측가능성), coverage(표정공간 평균 이격).
가설: 우울=repertoire 협착(PR↓·엔트로피↓·coverage↓).
질문: coupling에 정보 더하나 + 해석가능 mechanism?
CMDC/DAIC/LMVD. 결과: results/exp58_repertoire.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10; KST=6
def cov_of(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def PR(AU):  # participation ratio of z-scored AU covariance = 유효 표정차원
    Z=(AU-AU.mean(0))/(AU.std(0)+1e-6); C=np.cov(Z.T); w=np.linalg.eigvalsh(C); w=w[w>1e-9]
    return (w.sum()**2)/(np.square(w).sum()+1e-12)
def repertoire(AU):
    Z=(AU-AU.mean(0))/(AU.std(0)+1e-6)
    pr=PR(AU)
    n=min(len(Z),2000); Zs=Z[np.linspace(0,len(Z)-1,n).astype(int)]
    k=min(KST,max(2,len(np.unique(Zs,axis=0))))
    try:
        km=KMeans(k,n_init=3,random_state=0).fit(Zs); lab=km.labels_
        occ=np.bincount(lab,minlength=k)/len(lab); occ=occ[occ>0]
        occ_e=-(occ*np.log(occ)).sum()/np.log(k)  # 정규화 엔트로피
        # 전이 엔트로피
        tr=np.zeros((k,k))
        for a,b in zip(lab[:-1],lab[1:]): tr[a,b]+=1
        tr=tr/(tr.sum(1,keepdims=True)+1e-9); p0=np.bincount(lab,minlength=k)/len(lab)
        te=0
        for i in range(k):
            row=tr[i][tr[i]>0]; te+=p0[i]*(-(row*np.log(row)).sum())
        te/=np.log(k)
    except: occ_e=0;te=0
    cov=np.mean(np.linalg.norm(Zs-Zs.mean(0),axis=1))  # 평균 이격(coverage)
    return np.array([pr,occ_e,te,cov])
FN=['PR_유효차원','occ_entropy','trans_entropy','coverage']
def cv(COV,F,y,mode):
    COV=np.array(COV);F=np.nan_to_num(np.array(F));y=np.array(y);a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            if mode=='rep': X=F
            else:
                T=TangentSpace(metric='riemann').fit(COV[tr]);Xt=T.transform(COV)
                X=Xt if mode=='cov' else np.column_stack([Xt,F])
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def analyze(AUs, y, name):
    y=np.array(y); F=np.array([repertoire(a) for a in AUs]); COV=[cov_of(a) for a in AUs]
    print(f'\n===== {name} (n={len(y)}, 우울{y.sum()}) =====',flush=True)
    for j,nm in enumerate(FN):
        md,hc=np.nanmean(F[y==1,j]),np.nanmean(F[y==0,j])
        try:_,p=mannwhitneyu(F[y==1,j],F[y==0,j])
        except:p=1
        print(f'  {nm:13s} MDD={md:.3f} HC={hc:.3f} ({"↓협착" if md<hc else "↑"}우울) p={p:.3f}',flush=True)
    rr=cv(COV,F,y,'rep');rc=cv(COV,F,y,'cov');rb=cv(COV,F,y,'comb')
    print(f'  repertoire 단독 AUC={rr[0]:.3f}±{rr[1]:.3f}',flush=True)
    print(f'  정적 coupling  AUC={rc[0]:.3f}±{rc[1]:.3f}',flush=True)
    print(f'  결합           AUC={rb[0]:.3f}±{rb[1]:.3f}  {"★추가" if rb[0]>rc[0]+0.02 else "추가 없음"}',flush=True)
    return [name,len(y),rr[0],rc[0],rb[0]]
out=[]
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
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
out.append(analyze(A,Y,'CMDC'))
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
out.append(analyze(A,Y,'DAIC'))
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
    if len(fe)>=60: A.append(np.array(fe));Y.append(l)
out.append(analyze(A,Y,'LMVD'))
print('\n판정: (1)repertoire 협착 MDD<HC 유의 + (2)결합>coupling ≥2코퍼스 →',flush=True)
print('      "표정 repertoire 협착 지도"=해석가능 novelty 후보. 아니면 coupling에 포화 재확인.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp58_repertoire.csv','w') as f:
    f.write('corpus,n,repertoire_AUC,coupling_AUC,combined_AUC\n')
    for r in out:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp58_repertoire.csv',flush=True)
