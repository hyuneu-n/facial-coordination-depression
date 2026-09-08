"""
exp38 — Directional coupling (Granger causality) vs 정적 covariance.
가설: "누가 먼저 움직이나"(방향성 AU coupling)가 우울 신호를 담는다.
방법: anchor 구간 AU 시계열 → pairwise linear Granger(방향 d×d 비대칭) → per-window 계산 후 평균
      → 오프대각 벡터화 → 10seed StratifiedKFold logistic AUC + permutation.
비교: 정적 covariance(Riemannian tangent) baseline. + 결합(정적+방향).
데이터: DAIC-WOZ(ALL_neg anchor), CMDC(Q3+Q7).
결과: results/exp38_directional.csv
"""
import numpy as np, warnings, csv, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=list(range(10))

# ---------- Granger ----------
def _rss(D, y):
    D=np.column_stack([np.ones(len(y)), D])
    beta,_,_,_=np.linalg.lstsq(D,y,rcond=None)
    r=y-D@beta; return float(r@r)
def granger_dir(X, p=1):
    """X:(T,d) 정규화. return G(d,d): G[i,j]=i->j Granger(>=0)."""
    T,d=X.shape
    if T<=p+3: return None
    G=np.zeros((d,d))
    for j in range(d):
        y=X[p:,j]
        Yl=np.column_stack([X[p-1-k:T-1-k,j] for k in range(p)])
        rss_r=_rss(Yl,y)
        for i in range(d):
            if i==j: continue
            Xl=np.column_stack([X[p-1-k:T-1-k,i] for k in range(p)])
            rss_f=_rss(np.column_stack([Yl,Xl]),y)
            G[i,j]=np.log(max(rss_r,1e-9)/max(rss_f,1e-9))
    return G
def subj_granger(windows, p=1):
    Gs=[granger_dir((w-w.mean(0))/(w.std(0)+1e-6),p) for w in windows if len(w)>=15]
    Gs=[g for g in Gs if g is not None]
    return np.mean(Gs,0) if Gs else None

def offdiag(M):
    d=M.shape[0]; return M[~np.eye(d,dtype=bool)]

def auc_vec(F,y):
    F=np.nan_to_num(np.array(F));y=np.array(y);a=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
        for tr,te in skf.split(F,y):
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(F[tr],y[tr]);pb[te]=clf.decision_function(F[te])
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def auc_cov(C,y):
    C=np.array(C);y=np.array(y);a=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd);pb=np.zeros(len(y))
        for tr,te in skf.split(C,y):
            clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(C[tr],y[tr]);pb[te]=clf.decision_function(C[te])
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def perm_p(F,y,obs,vec=True):
    y=np.array(y);rng=np.random.RandomState(0);null=[]
    for _ in range(100):
        yp=rng.permutation(y)
        m,_=auc_vec(F,yp) if vec else auc_cov(F,yp)
        null.append(m)
    return (np.sum(np.array(null)>=obs)+1)/101

def run(windows_by_subj, y, name):
    subs=list(windows_by_subj.keys()); y=np.array(y)
    G=[subj_granger(windows_by_subj[s]) for s in subs]
    cov=[]
    for s in subs:
        seg=np.vstack(windows_by_subj[s]); seg=(seg-seg.mean(0))/(seg.std(0)+1e-6); c,_=ledoit_wolf(seg); cov.append(c)
    keep=[i for i,g in enumerate(G) if g is not None]
    subs=[subs[i] for i in keep]; y=y[keep]; G=[G[i] for i in keep]; cov=[cov[i] for i in keep]
    Fdir=[offdiag(g) for g in G]
    # 방향성 비대칭 정도 (net flow)
    dir_auc=auc_vec(Fdir,y)
    cov_auc=auc_cov(cov,y)
    # 결합: tangent(cov) + dir
    ts=TangentSpace(metric='riemann'); T=ts.fit_transform(np.array(cov))
    Fcomb=[np.concatenate([T[i],Fdir[i]]) for i in range(len(y))]
    comb_auc=auc_vec(Fcomb,y)
    pdir=perm_p(Fdir,y,dir_auc[0],vec=True)
    print(f'\n=== {name} n={len(y)} 우울{int(y.sum())} ===',flush=True)
    print(f'  정적 covariance(baseline)  AUC={cov_auc[0]:.3f}±{cov_auc[1]:.3f}',flush=True)
    print(f'  방향성 Granger(exp38)       AUC={dir_auc[0]:.3f}±{dir_auc[1]:.3f}  perm_p={pdir:.3f}',flush=True)
    print(f'  결합(정적+방향)             AUC={comb_auc[0]:.3f}±{comb_auc[1]:.3f}',flush=True)
    # 방향성 MDD vs HC: 평균 net-flow 차이 top
    Gm=np.mean([G[i] for i in range(len(y)) if y[i]==1],0)-np.mean([G[i] for i in range(len(y)) if y[i]==0],0)
    d=Gm.shape[0]; iu=[(i,j) for i in range(d) for j in range(d) if i!=j]
    vals=[(abs(Gm[i,j]),i,j,Gm[i,j]) for i,j in iu]; vals.sort(reverse=True)
    print(f'  MDD-HC 방향 차이 top3(i->j, Δ):',[(f'AU{a}->AU{b}',round(v,3)) for _,a,b,v in vals[:3]],flush=True)
    return name,len(y),int(y.sum()),cov_auc[0],dir_auc[0],pdir,comb_auc[0]

results=[]
# ===== CMDC =====
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]
    except:return None
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if len(fe)>=15 else None
Wc={};yc=[]
for s,l in cl.items():
    ws_=[cq(s,q) for q in [3,7]]; ws_=[w for w in ws_ if w is not None]
    if len(ws_)==2: Wc[s]=ws_; yc.append(l)
results.append(run(Wc,yc,'CMDC(anchor Q3+7)'))

# ===== DAIC =====
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
NEG=['feel_lately','depression_diagnosed','feelguilty','regret','feelbadly','last_argument','control_temper']
def dseries(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists():return None,None
    h=[x.strip() for x in open(p).readline().split(',')];ti,oi=h.index('timestamp'),h.index('success');ai=[h.index(c) for c in AUd]
    ts,fe=[],[]
    for ln in open(p).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1:continue
            ts.append(float(v[ti]));fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(ts),np.array(fe)
def dtrans(pid):
    p=D/f'{pid}_TRANSCRIPT.csv';r_=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try:r_.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except:pass
    return r_
Wd={};yd=[]
for pid,l in dl.items():
    ts,au=dseries(pid)
    if ts is None:continue
    wins=[]
    for st,sp,spk,val in dtrans(pid):
        if spk=='Ellie' and any(val.startswith(t) for t in NEG):
            m=(ts>=sp)&(ts<sp+8.0)
            if m.sum()>=15:wins.append(au[m])
    if wins: Wd[pid]=wins; yd.append(l)
results.append(run(Wd,yd,'DAIC(anchor ALLneg)'))

with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp38_directional.csv','w') as f:
    f.write('dataset,n,dep,cov_AUC,dir_AUC,dir_perm_p,comb_AUC\n')
    for nm,n,dd,c,di,p,cb in results:f.write(f'{nm},{n},{dd},{c:.4f},{di:.4f},{p:.4f},{cb:.4f}\n')
print('\nDONE → exp38_directional.csv',flush=True)
