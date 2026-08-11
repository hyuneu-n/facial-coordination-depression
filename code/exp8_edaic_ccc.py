"""
exp8 — E-DAIC(275명, OpenFace2.0) 스모크. 지표: CCC + MAE (심각도 회귀) + AUC(보조).
앵커 아직 없음(transcript 질문자 X) → 전체 인터뷰 AU로 규모/품질 효과 확인.
저차원성 지수도 회귀로 테스트.
"""
import numpy as np, csv, glob, warnings
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
warnings.filterwarnings('ignore')

E=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/E-DAIC')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r',
    'AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
SEEDS=[42,1,2,3,4]

def ccc(y,yp):
    y,yp=np.asarray(y,float),np.asarray(yp,float)
    my,mp=y.mean(),yp.mean(); vy,vp=y.var(),yp.var(); cov=((y-my)*(yp-mp)).mean()
    return 2*cov/(vy+vp+(my-mp)**2+1e-9)

def labels():
    lab={}
    for f in ['train_split.csv','dev_split.csv','test_split.csv']:
        p=E/'labels'/f
        if p.exists():
            for r in csv.DictReader(open(p)):
                pid=r['Participant_ID'].strip()
                sc=r.get('PHQ_Score') or r.get('PHQ8_Score')
                bn=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
                if sc not in (None,''): lab[pid]=(float(sc),int(float(bn)) if bn not in (None,'') else int(float(sc)>=10))
    return lab

def au_series(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*AUs.csv'))
    if not fs: return None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try: oi=h.index('success'); ai=[h.index(c) for c in AU]
    except: return None
    fe=[]
    with open(fs[0]) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(fe) if len(fe)>10 else None

lab=labels(); data={}
for pid in lab:
    au=au_series(pid)
    if au is not None: data[pid]=au
score=np.array([lab[p][0] for p in data]); binr=np.array([lab[p][1] for p in data])
print(f'E-DAIC 로드: {len(data)}명, PHQ평균{score.mean():.1f}, 우울(≥10){int((score>=10).sum())}\n',flush=True)

def statfeat(au): return np.concatenate([au.mean(0),au.std(0)])
def lowdim(au):
    X=au-au.mean(0,keepdims=True); s=np.linalg.svd(X,compute_uv=False); s=s[s>1e-8]
    if len(s)<2: return np.zeros(3)
    p=s/s.sum(); s2=s**2
    return np.array([p[0], (s2.sum()**2)/((s2**2).sum()), np.exp(-(p*np.log(p+1e-12)).sum())])

def evaluate(featfn,name):
    pids=list(data); X=np.array([featfn(data[p]) for p in pids])
    ys=np.array([lab[p][0] for p in pids]); yb=np.array([lab[p][1] for p in pids])
    cccs,maes,aucs=[],[],[]
    for s in SEEDS:
        kf=KFold(5,shuffle=True,random_state=s)
        pr=np.zeros_like(ys,dtype=float)
        for tr,te in kf.split(X):
            sc=StandardScaler().fit(X[tr]); m=Ridge(alpha=10).fit(sc.transform(X[tr]),ys[tr])
            pr[te]=m.predict(sc.transform(X[te]))
        cccs.append(ccc(ys,pr)); maes.append(mean_absolute_error(ys,pr))
        # 이진 AUC (회귀예측을 점수로)
        try: aucs.append(roc_auc_score(yb,pr))
        except: pass
    print(f'  [{name}] CCC={np.mean(cccs):.3f}  MAE={np.mean(maes):.2f}  AUC={np.mean(aucs):.3f}',flush=True)

print('=== E-DAIC 전체인터뷰 (앵커無) — 지표 CCC/MAE/AUC ===',flush=True)
evaluate(statfeat,'AU 통계')
evaluate(lowdim,'저차원성')
evaluate(lambda a:np.concatenate([statfeat(a),lowdim(a)]),'AU통계+저차원')
print(f'\n(참고: DAIC 앵커 AUC 0.6대 / AVEC 논문 MAE 5~7)',flush=True)
