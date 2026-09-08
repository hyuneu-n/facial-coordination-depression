"""
exp73 — 멀티모달 강화(우리 lane): 얼굴(coupling+expressivity) + 음성(coupling+prosodic).
남들 딥퓨전 아니라, 우리 해석가능 coupling+expressivity를 음성에도 적용해 결합.
질문: 음성 얹으면 얼굴-단독 대비 within-corpus AUC 오르나? (LMVD·D-Vlog 우리 finding 강한 곳)
결과: results/exp73_multimodal.csv
"""
import numpy as np, warnings, csv, glob
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10; K=20
def cov_pca(X,pca):
    Z=pca.transform(X); c,_=ledoit_wolf(Z); return c+1e-3*np.eye(Z.shape[1])
def expr_vec(X):  # 활성/변동
    d=np.linalg.norm(np.diff(X,axis=0),axis=1) if len(X)>1 else np.array([0.0])
    return np.array([X.std(0).mean(), np.abs(X-X.mean(0)).mean(), d.mean(), d.std()])
def run_cv(feats_by_mode, y):
    """feats_by_mode: dict mode-> (COV list or None, EXP array). build tangent per fold."""
    y=np.array(y); res={}
    modes=list(feats_by_mode)
    def buildX(tr, sel):
        parts=[]
        for m in sel:
            COV,EXP=feats_by_mode[m]
            if COV is not None:
                T=TangentSpace(metric='riemann').fit(np.array(COV)[tr]);parts.append(T.transform(np.array(COV)))
            if EXP is not None: parts.append(EXP)
        return np.column_stack(parts)
    for sel,nm in [(['face'],'face'),(['aud'],'audio'),(['face','aud'],'face+audio')]:
        a=[]
        for s in range(SEEDS):
            skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
            for tr,te in skf.split(np.zeros(len(y)),y):
                X=np.nan_to_num(buildX(tr,sel))
                sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
                pb[te]=cl.decision_function(sc.transform(X[te]))
            a.append(roc_auc_score(y,pb))
        res[nm]=(np.mean(a),np.std(a))
    return res
# ---- LMVD ----
def load_lmvd():
    V=B/'LMVD/extracted/Video_feature'; Au=B/'LMVD/extracted/Audio_feature'
    AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
    def lab(i):
        if (1<=i<=601) or (1117<=i<=1423): return 1
        if (602<=i<=1116) or (1425<=i<=1824): return 0
        return None
    facf=[];audf=[];y=[]
    for f in sorted(glob.glob(str(V/'*.csv'))):
        sid=Path(f).stem; l=lab(int(sid))
        if l is None:continue
        ap=Au/f'{sid}.npy'
        if not ap.exists():continue
        h=[x.strip() for x in open(f).readline().split(',')]
        try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AU]
        except:continue
        fe=[]
        for ln in open(f).readlines()[1:]:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
                fe.append([float(v[i]) for i in ai])
            except:pass
        if len(fe)<60:continue
        try:au=np.nan_to_num(np.load(ap).astype(float))
        except:continue
        if au.ndim!=2 or len(au)<10:continue
        facf.append(np.nan_to_num(np.array(fe)));audf.append(au);y.append(l)
    return facf,audf,y
# ---- D-Vlog ----
def load_dvlog():
    R=B/'D-Vlog'
    def nf(V):
        T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
        sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
    facf=[];audf=[];y=[]
    for r in csv.DictReader(open(R/'labels.csv')):
        idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
        if not(fv.exists() and fa.exists()):continue
        try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
        except:continue
        if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136:continue
        if Au.ndim!=2 or len(Au)<10:continue
        facf.append(nf(V));audf.append(Au);y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
    return facf,audf,y
def analyze(facf,audf,y,name,facePCA):
    y=np.array(y); print(f'\n===== {name} (n={len(y)}, 우울{int(y.sum())}) =====',flush=True)
    # 얼굴 PCA fit
    fp=PCA(min(K,facf[0].shape[1])).fit(np.vstack([f[::3] for f in facf]))
    ap=PCA(min(K,audf[0].shape[1])).fit(np.vstack([a[::3] for a in audf]))
    faceCOV=[cov_pca(f,fp) for f in facf]; faceEXP=np.array([expr_vec(f) for f in facf])
    audCOV=[cov_pca(a,ap) for a in audf]; audEXP=np.array([expr_vec(a) for a in audf])
    feats={'face':(faceCOV,faceEXP),'aud':(audCOV,audEXP)}
    res=run_cv(feats,y)
    for k in ['face','audio','face+audio']:
        print(f'  {k:12s} AUC={res[k][0]:.3f}±{res[k][1]:.3f}',flush=True)
    boost=res['face+audio'][0]-res['face'][0]
    print(f'  → 음성 얹은 부스트 Δ={boost:+.3f} {"★" if boost>0.02 else ""}',flush=True)
    return [name,len(y),res['face'][0],res['audio'][0],res['face+audio'][0]]
out=[]
print('LMVD 로딩...',flush=True); f,a,y=load_lmvd(); out.append(analyze(f,a,y,'LMVD',True))
print('D-Vlog 로딩...',flush=True); f,a,y=load_dvlog(); out.append(analyze(f,a,y,'D-Vlog',True))
print('\n판정: face+audio가 face 단독 대비 +0.02↑ → 멀티모달 강화 성립(우리 해석가능 coupling 틀 유지).',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp73_multimodal.csv','w') as f_:
    f_.write('corpus,n,face,audio,face_audio\n')
    for r in out:f_.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp73_multimodal.csv',flush=True)
