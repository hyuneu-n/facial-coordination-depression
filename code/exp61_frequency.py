"""
exp61 — idea(PDF2 STSFF-Net 근거): 주파수영역 AU 특징. 아직 안 해본 표현.
정적 covariance가 버리는 '시간척도별 에너지 분포'를 잡음. 가설: 우울=flat affect/정신운동
 → 고주파 표정 dynamics 에너지↓, 저주파 비중↑, 스펙트럼 엔트로피↓.
특징: AU별 band energy ratio(vlow/low/mid/high) + spectral centroid + entropy.
질문: freq 단독? coupling에 추가? 기전(고주파↓)? CMDC/DAIC/E-DAIC/LMVD.
결과: results/exp61_frequency.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10; FPS=30
BANDS=[(0,0.2),(0.2,0.5),(0.5,1.5),(1.5,5.0)]
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
def cov_of(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def spectral_feats(AU):
    Z=(AU-AU.mean(0))/(AU.std(0)+1e-6); T=len(Z); freqs=np.fft.rfftfreq(T,d=1.0/FPS)
    feats=[]; hi_all=[]
    for c in range(Z.shape[1]):
        ps=np.abs(np.fft.rfft(Z[:,c]))**2; tot=ps.sum()+1e-9
        ratios=[ps[(freqs>=lo)&(freqs<hi)].sum()/tot for lo,hi in BANDS]
        cen=(freqs*ps).sum()/tot
        pn=ps/tot; ent=-(pn[pn>0]*np.log(pn[pn>0])).sum()/np.log(len(pn))
        feats.extend(ratios+[cen,ent]); hi_all.append(ratios[3])  # high-band ratio
    return np.array(feats), np.mean(hi_all)
def cv(COV,F,y,mode):
    COV=np.array(COV);F=np.nan_to_num(np.array(F));y=np.array(y);a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            if mode=='freq':
                k=min(20,len(tr)-1,F.shape[1]);pc=PCA(k).fit(F[tr]);X=pc.transform(F)
            else:
                T=TangentSpace(metric='riemann').fit(COV[tr]);Xt=T.transform(COV)
                if mode=='cov':X=Xt
                else:
                    k=min(20,len(tr)-1,F.shape[1]);pc=PCA(k).fit(F[tr]);X=np.column_stack([Xt,pc.transform(F)])
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def analyze(AUs,y,name):
    y=np.array(y); F=[];HI=[];COV=[]
    for au in AUs:
        f,hi=spectral_feats(au); F.append(f);HI.append(hi);COV.append(cov_of(au))
    F=np.array(F);HI=np.array(HI)
    md,hc=HI[y==1].mean(),HI[y==0].mean()
    try:_,p=mannwhitneyu(HI[y==1],HI[y==0])
    except:p=1
    print(f'\n===== {name} (n={len(y)}, 우울{y.sum()}) =====',flush=True)
    print(f'  고주파에너지비 MDD={md:.4f} HC={hc:.4f} ({"↓우울(flat affect)" if md<hc else "↑"}) p={p:.3f}',flush=True)
    rf=cv(COV,F,y,'freq');rc=cv(COV,F,y,'cov');rb=cv(COV,F,y,'comb')
    print(f'  주파수 단독   AUC={rf[0]:.3f}±{rf[1]:.3f}',flush=True)
    print(f'  coupling      AUC={rc[0]:.3f}±{rc[1]:.3f}',flush=True)
    print(f'  결합          AUC={rb[0]:.3f}±{rb[1]:.3f}  {"★추가" if rb[0]>rc[0]+0.02 else "추가 없음"}',flush=True)
    return [name,len(y),rf[0],rc[0],rb[0]]
out=[]
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
    if parts and sum(len(p) for p in parts)>=120: A.append(np.vstack(parts));Y.append(l)
out.append(analyze(A,Y,'CMDC'))
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
    if au is not None and len(au)>=120: A.append(au);Y.append(l)
out.append(analyze(A,Y,'DAIC'))
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
    return np.array(fe) if len(fe)>=120 else None
A=[];Y=[]
for pid,l in el.items():
    au=eau(pid)
    if au is not None: A.append(au);Y.append(l)
out.append(analyze(A,Y,'E-DAIC'))
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
    if len(fe)>=120: A.append(np.array(fe));Y.append(l)
out.append(analyze(A,Y,'LMVD'))
print('\n판정: freq 단독>coupling or 결합>coupling(+0.02) ≥2코퍼스 or 고주파↓ 일관 → 주파수 novelty 후보.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp61_frequency.csv','w') as f:
    f.write('corpus,n,freq_only,coupling,combined\n')
    for r in [x for x in out if x]:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp61_frequency.csv',flush=True)
