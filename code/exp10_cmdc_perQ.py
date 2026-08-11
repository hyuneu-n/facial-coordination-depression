"""
exp10 — CMDC Q별 판별력 스캔. 어느 질문(Q) 구간이 우울 신호 강한가 = 앵커 찾기.
질문 내용 몰라도 데이터가 알려줌. 각 Q 단독 + 답변 텍스트 샘플로 질문성격 추정.
지표 CCC/MAE/AUC.
"""
import numpy as np, warnings, openpyxl
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
warnings.filterwarnings('ignore')
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
AU_R=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r',
      'AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
SEEDS=[42,1,2,3,4]
def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean();vy,vp=y.var(),yp.var();cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0])
iID,iMDD=hd.index('ID'),hd.index('MDD'); iP=[hd.index(f'PHQ-{i}') for i in range(1,10)]
lab={}
for r in rows[1:]:
    if r[iID] is None: continue
    try: tot=sum(int(r[i]) for i in iP if r[i] is not None)
    except: continue
    lab[str(r[iID]).strip()]=(tot,int(r[iMDD]))

def au_q(subj,q):
    f=C/subj/f'Q{q}.csv'
    if not f.exists(): return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try: oi=h.index('success'); ai=[h.index(c) for c in AU_R]
    except: return None
    fe=[]
    with open(f) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(fe) if len(fe)>=5 else None

def statfeat(au): return np.concatenate([au.mean(0),au.std(0)])

print(f"{'Q':>3} {'n':>4} {'CCC':>7} {'MAE':>6} {'AUC':>7}  답변샘플",flush=True)
print('-'*70,flush=True)
best=[]
for q in range(1,13):
    subs=[s for s in lab if au_q(s,q) is not None]
    if len(subs)<25: continue
    X=np.array([statfeat(au_q(s,q)) for s in subs])
    ys=np.array([lab[s][0] for s in subs]); yb=np.array([lab[s][1] for s in subs])
    cccs,maes,aucs=[],[],[]
    for sd in SEEDS:
        kf=KFold(5,shuffle=True,random_state=sd); pr=np.zeros(len(ys))
        for tr,te in kf.split(X):
            sc=StandardScaler().fit(X[tr]); m=Ridge(alpha=10).fit(sc.transform(X[tr]),ys[tr]); pr[te]=m.predict(sc.transform(X[te]))
        cccs.append(ccc(ys,pr)); maes.append(mean_absolute_error(ys,pr))
        try: aucs.append(roc_auc_score(yb,pr))
        except: pass
    # 답변 샘플 (질문 성격 추정)
    txt=''
    tf=C/subs[0]/f'Q{q}.txt'
    if tf.exists(): txt=open(tf,encoding='utf-8').read().strip()[:30]
    cm,mm,am=np.mean(cccs),np.mean(maes),np.mean(aucs)
    best.append((q,cm,am))
    print(f"{q:>3} {len(subs):>4} {cm:>7.3f} {mm:>6.2f} {am:>7.3f}  {txt}",flush=True)

best.sort(key=lambda r:-r[1])
print(f"\n[최고 판별 질문] CCC 기준: Q{best[0][0]} (CCC={best[0][1]:.3f}), Q{best[1][0]}, Q{best[2][0]}",flush=True)
print(f"[AUC 기준] " + ", ".join(f"Q{q}({a:.2f})" for q,c,a in sorted(best,key=lambda r:-r[2])[:3]),flush=True)
