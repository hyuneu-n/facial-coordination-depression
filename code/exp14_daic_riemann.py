"""
exp14 — DAIC에 Riemannian (교차 재현). CMDC서 AUC 0.88 → DAIC서도?
DAIC 긍정앵커 구간 AU → 공분산 SPD → 접공간+로지스틱/Ridge.
앵커 有(긍정) vs 無(전체) 대조. baseline: DAIC ST-GCN 앵커 AUC0.63.
"""
import numpy as np, csv, warnings
from pathlib import Path
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error
warnings.filterwarnings('ignore')
D=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
POST=6.0; SEEDS=[42,1,2,3,4]
POS=['last time you felt really happy','really happy','proud','enjoy','felt really happy']
def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean();vy,vp=y.var(),yp.var();cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)
def labels():
    lab={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)):
                lab[r['Participant_ID'].strip()]=(float(r['PHQ8_Score']),int(float(r['PHQ8_Binary'])))
    return lab
def transcript(pid):
    p=D/f'{pid}_TRANSCRIPT.csv'; rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try: rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except: pass
    return rows
def anchors(rows,keys): return [sp for st,sp,spk,val in rows if spk=='Ellie' and any(k in val for k in keys)]
def au_series(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists(): return None,None
    h=[x.strip() for x in open(p).readline().split(',')]
    ti,oi=h.index('timestamp'),h.index('success'); ai=[h.index(c) for c in AU]
    ts,fe=[],[]
    with open(p) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                ts.append(float(v[ti])); fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(ts),np.array(fe)
lab=labels()
def get_cov(pid,mode):
    ts,au=au_series(pid)
    if ts is None: return None
    if mode=='all': seg=au
    else:
        segs=[au[(ts>=a)&(ts<a+POST)] for a in anchors(transcript(pid),POS)]
        segs=[s for s in segs if len(s)>=6]
        if not segs: return None
        seg=np.vstack(segs)
    if len(seg)<10: return None
    c=np.cov(seg.T)+1e-6*np.eye(len(AU))
    return c
def run(mode,name):
    covs,ys,yb=[],[],[]
    for pid,(sc,bn) in lab.items():
        c=get_cov(pid,mode)
        if c is not None: covs.append(c); ys.append(sc); yb.append(bn)
    covs=np.array(covs); ys=np.array(ys); yb=np.array(yb)
    aucs,cccs,maes=[],[],[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(yb),float)
        for tr,te in skf.split(covs,yb):
            clf=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(covs[tr],yb[tr]); pb[te]=clf.decision_function(covs[te])
        aucs.append(roc_auc_score(yb,pb))
        kf=KFold(5,shuffle=True,random_state=sd); pr=np.zeros(len(ys),float)
        for tr,te in kf.split(covs):
            reg=make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),Ridge(alpha=10))
            reg.fit(covs[tr],ys[tr]); pr[te]=reg.predict(covs[te])
        cccs.append(ccc(ys,pr)); maes.append(mean_absolute_error(ys,pr))
    print(f'  [{name:16s}] n={len(ys)} CCC={np.mean(cccs):.3f} MAE={np.mean(maes):.2f} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}',flush=True)
print('=== DAIC Riemannian 교차 재현 (baseline ST-GCN 앵커 0.63) ===',flush=True)
run('pos','앵커 긍정')
run('all','전체 인터뷰')
