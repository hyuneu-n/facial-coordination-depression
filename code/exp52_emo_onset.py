"""
exp52 — 감정(부정)-onset-anchored precursor. ★상류 교정★
지금까지 event=generic 모션peak → 정적특징 못 넘음. 북극성은 '부정 표정 짓기 前'.
재정의: valence(t)=z(AU6+AU12=긍정) - z(AU4+AU15+AU1=부정/슬픔).
  부정 onset = valence 국소최소(표정이 부정으로 꺾이는 순간).
검증: 부정-onset 직전[t*-3s,-0.5s] coupling이 whole-session/긍정-onset보다 우울 잘 잡나?
  → 그렇다면 '부정표정 짓기 전 협응 붕괴'라는 goal-정합·precursor-특이 신규 신호.
CMDC/DAIC(임상) + LMVD(대형). 결과: results/exp52_emo_onset.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); PRE0=90; PRE1=15
def valence(AU, idx):
    def z(x):return (x-x.mean())/(x.std()+1e-6)
    pos=sum(AU[:,idx[a]] for a in ['AU06_r','AU12_r'] if a in idx)
    neg=sum(AU[:,idx[a]] for a in ['AU04_r','AU15_r','AU01_r'] if a in idx)
    return z(pos)-z(neg)
def onset_cov(AU, idx, sign):
    """sign=-1 부정onset, +1 긍정onset. 해당 onset 직전창 프레임 모아 covariance"""
    val=valence(AU,idx); sig=-val if sign<0 else val
    mad=np.median(np.abs(sig-np.median(sig)))+1e-6
    ev,_=find_peaks(sig, prominence=1.0*mad, distance=45)
    ev=[e for e in ev if e-PRE0>=0]
    if len(ev)<3: return None
    frames=np.vstack([AU[e-PRE0:e-PRE1] for e in ev])
    if len(frames)<30: return None
    seg=(frames-frames.mean(0))/(frames.std(0)+1e-6)
    c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def whole_cov(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def cv_auc(COV, y):
    COV=np.array(COV); y=np.array(y); a=[]
    for s in range(10):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(COV,y):
            t=TangentSpace(metric='riemann').fit(COV[tr]);X=t.transform(COV)
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def analyze(subjAU, Y, AUlist, name):
    idx={a:i for i,a in enumerate(AUlist)}
    W=[];N=[];P=[];yw=[];yn=[];yp=[]
    for au,l in zip(subjAU,Y):
        if len(au)<PRE0+30: continue
        W.append(whole_cov(au));yw.append(l)
        cn=onset_cov(au,idx,-1)
        if cn is not None: N.append(cn);yn.append(l)
        cp=onset_cov(au,idx,+1)
        if cp is not None: P.append(cp);yp.append(l)
    rw=cv_auc(W,yw); rn=cv_auc(N,yn) if len(set(yn))>1 and len(yn)>15 else (float('nan'),0)
    rp=cv_auc(P,yp) if len(set(yp))>1 and len(yp)>15 else (float('nan'),0)
    print(f'\n===== {name} =====',flush=True)
    print(f'  whole-session coupling  AUC={rw[0]:.3f}±{rw[1]:.3f} (n={len(yw)})',flush=True)
    print(f'  부정-onset 직전 coupling AUC={rn[0]:.3f}±{rn[1]:.3f} (n={len(yn)})  {"★개선" if rn[0]>rw[0]+0.02 else ""}',flush=True)
    print(f'  긍정-onset 직전 coupling AUC={rp[0]:.3f}±{rp[1]:.3f} (n={len(yp)})',flush=True)
    return [name,rw[0],rn[0],rp[0],len(yw),len(yn),len(yp)]
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
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
sA=[];sy=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts: sA.append(np.vstack(parts));sy.append(l)
out.append(analyze(sA,sy,AUc,'CMDC'))
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
sA=[];sy=[]
for pid,l in dl.items():
    au=dau(pid)
    if au is not None: sA.append(au);sy.append(l)
out.append(analyze(sA,sy,AUd,'DAIC'))
# LMVD
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423): return 1
    if (602<=i<=1116) or (1425<=i<=1824): return 0
    return None
sA=[];sy=[]
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
    if len(fe)>=PRE0+30: sA.append(np.array(fe));sy.append(l)
out.append(analyze(sA,sy,AUc,'LMVD'))
print('\n판정: 부정-onset 직전 coupling AUC > whole & > 긍정-onset 이 여러 코퍼스서 일관 →',flush=True)
print('      "부정 표정 짓기 전 협응 붕괴"=goal정합·precursor-특이 신규 신호(novelty 후보).',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp52_emo_onset.csv','w') as f:
    f.write('corpus,whole_AUC,neg_onset_AUC,pos_onset_AUC,n_whole,n_neg,n_pos\n')
    for r in out:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp52_emo_onset.csv',flush=True)
