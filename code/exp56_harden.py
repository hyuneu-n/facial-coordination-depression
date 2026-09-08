"""
exp56 — exp55 굳히기 3종: (1)E-DAIC 4번째 코퍼스 (2)bootstrap CI + Δ유의성
  (3)약한쌍 진단(domain-gap=domain mean 간 Riemann거리).
프로토콜: 순수 zero-shot + unsupervised 정렬(recenter+scale, 타깃 라벨 0개).
  분류기=source만 학습, 정렬=source+target 무라벨. 4코퍼스 12쌍.
결과: results/exp56_transfer.csv + exp56_gap.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.covariance import ledoit_wolf
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import encode_domains, TLCenter, TLScale
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); RNG=np.random.RandomState(0); NBOOT=1000
COM=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
def cov_mat(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def base_clf():
    return make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
def align(Xs,Xt):  # unsupervised recenter+scale, 타깃 라벨 0
    Xf=np.concatenate([Xs,Xt]); df=np.array(['src']*len(Xs)+['tgt']*len(Xt)); yf=np.zeros(len(Xf),int)
    Xe,ye=encode_domains(Xf,yf,df)
    tl=make_pipeline(TLCenter(target_domain='tgt'),TLScale(target_domain='tgt',centered_data=True)); tl.fit(Xe,ye)
    Xal=tl.transform(Xe); return Xal[:len(Xs)],Xal[len(Xs):]
def boot_ci(y,pb,pb0):
    y=np.array(y); n=len(y); bA=[];bD=[]
    for _ in range(NBOOT):
        idx=RNG.randint(0,n,n)
        if len(np.unique(y[idx]))<2: continue
        aA=roc_auc_score(y[idx],pb[idx]); a0=roc_auc_score(y[idx],pb0[idx]); bA.append(aA);bD.append(aA-a0)
    return np.percentile(bA,[2.5,97.5]),np.percentile(bD,[2.5,97.5])
def run_pair(src,tgt,data,out,gaprows):
    Xs,ys=data[src];Xt,yt=data[tgt]; Xs=np.array(Xs);Xt=np.array(Xt);ys=np.array(ys);yt=np.array(yt)
    pb0=base_clf().fit(Xs,ys).predict_proba(Xt)[:,1]
    Xs_al,Xt_al=align(Xs,Xt); pbA=base_clf().fit(Xs_al,ys).predict_proba(Xt_al)[:,1]
    a0=roc_auc_score(yt,pb0); aA=roc_auc_score(yt,pbA)
    ciA,ciD=boot_ci(yt,pbA,pb0); sig='★유의' if ciD[0]>0 else 'n.s.'
    # domain gap
    Ms=mean_riemann(Xs);Mt=mean_riemann(Xt); gap=distance_riemann(Ms,Mt)
    print(f'  {src:6s}→{tgt:6s} zero{a0:.3f} → align {aA:.3f} [{ciA[0]:.3f},{ciA[1]:.3f}]  Δ{aA-a0:+.3f}[{ciD[0]:+.3f},{ciD[1]:+.3f}] {sig}  gap={gap:.2f}',flush=True)
    out.append([src,tgt,len(ys),len(yt),a0,aA,ciA[0],ciA[1],aA-a0,ciD[0],ciD[1],gap])
    gaprows.append([src,tgt,gap,aA-a0,aA])
# ---- loaders (공통 14 AU) ----
def _cov_list(mats): return [cov_mat(m) for m in mats]
def load_cmdc():
    C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
    rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
    cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
    def cq(s,q):
        f=C/s/f'Q{q}.csv'
        if not f.exists():return None
        h=[x.strip() for x in open(f).readline().split(',')]
        try:oi=h.index('success');ai=[h.index(c) for c in COM]
        except:return None
        fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
        return np.array(fe) if fe else None
    X=[];Y=[]
    for s,l in cl.items():
        parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
        if parts and sum(len(p) for p in parts)>=60: X.append(cov_mat(np.vstack(parts)));Y.append(l)
    return X,Y
def load_daic():
    D=B/'DAIC_WOZ';dl={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
    def dau(pid):
        p=D/f'{pid}_CLNF_AUs.txt'
        if not p.exists():return None
        h=[x.strip() for x in open(p).readline().split(',')];oi=h.index('success')
        try:ai=[h.index(c) for c in COM]
        except:return None
        fe=[]
        for ln in open(p).readlines()[1:]:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1:continue
                fe.append([float(v[i]) for i in ai])
            except:pass
        return np.array(fe) if fe else None
    X=[];Y=[]
    for pid,l in dl.items():
        au=dau(pid)
        if au is not None and len(au)>=60: X.append(cov_mat(au));Y.append(l)
    return X,Y
def load_edaic():
    E=B/'E-DAIC';el={}
    for f in ['train_split.csv','dev_split.csv','test_split.csv']:
        p=E/'labels'/f
        if p.exists():
            for r in csv.DictReader(open(p)):
                pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
                if b not in(None,''):el[pid]=int(float(b))
    def eau(pid):
        fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
        if not fs:return None
        h=[x.strip() for x in open(fs[0]).readline().split(',')]
        try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in COM]
        except:return None
        fe=[]
        for ln in open(fs[0]).readlines()[1:]:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
                fe.append([float(v[i]) for i in ai])
            except:pass
        return np.array(fe) if fe else None
    X=[];Y=[]
    for pid,l in el.items():
        au=eau(pid)
        if au is not None and len(au)>=60: X.append(cov_mat(au));Y.append(l)
    return X,Y
def load_lmvd():
    V=B/'LMVD/extracted/Video_feature'
    def lab(i):
        if (1<=i<=601) or (1117<=i<=1423): return 1
        if (602<=i<=1116) or (1425<=i<=1824): return 0
        return None
    X=[];Y=[]
    for f in sorted(glob.glob(str(V/'*.csv'))):
        l=lab(int(Path(f).stem))
        if l is None:continue
        h=[x.strip() for x in open(f).readline().split(',')]
        try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in COM]
        except:continue
        fe=[]
        for ln in open(f).readlines()[1:]:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
                fe.append([float(v[i]) for i in ai])
            except:pass
        if len(fe)>=60: X.append(cov_mat(np.array(fe)));Y.append(l)
    return X,Y
print('로딩 CMDC/DAIC/E-DAIC...',flush=True)
data={'CMDC':load_cmdc(),'DAIC':load_daic(),'EDAIC':load_edaic()}
for k in ['CMDC','DAIC','EDAIC']: print(f'  {k}: n={len(data[k][1])} 우울{sum(data[k][1])}',flush=True)
out=[];gaprows=[]
print('\n--- 임상 3코퍼스 상호 전이 (E-DAIC⊇DAIC라 DAIC↔EDAIC는 참고) ---',flush=True)
clin=['CMDC','DAIC','EDAIC']
for s in clin:
    for t in clin:
        if s!=t: run_pair(s,t,data,out,gaprows)
print('\n로딩 LMVD...',flush=True)
data['LMVD']=load_lmvd(); print(f'  LMVD: n={len(data["LMVD"][1])}',flush=True)
print('\n--- LMVD(대형 야생) 관련 전이 ---',flush=True)
for s,t in [('CMDC','LMVD'),('DAIC','LMVD'),('EDAIC','LMVD'),('LMVD','CMDC'),('LMVD','DAIC'),('LMVD','EDAIC')]:
    run_pair(s,t,data,out,gaprows)
# 상관: gap vs Δ
import numpy as _np
G=_np.array([r[2] for r in gaprows]);Dv=_np.array([r[3] for r in gaprows])
if len(G)>2:
    cc=_np.corrcoef(G,Dv)[0,1]; print(f'\n[진단] domain-gap vs 정렬이득Δ 상관 r={cc:.2f} (음수면 gap클수록 이득작음=정렬한계)',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp56_transfer.csv','w') as f:
    f.write('src,tgt,n_src,n_tgt,zeroshot,align,ci_lo,ci_hi,delta,d_lo,d_hi,gap\n')
    for r in out:f.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp56_gap.csv','w') as f:
    f.write('src,tgt,gap,delta,align_auc\n')
    for r in gaprows:f.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('\nDONE → exp56_transfer.csv + exp56_gap.csv',flush=True)
