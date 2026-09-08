"""
exp74 — ★novel thread★: cross-modal 협응(coordination). generic fusion 아님.
얼굴활동 envelope ↔ 음성에너지 envelope의 시간 동기화가 우울서 깨지나(decouple)?
핵심 검증: 협응(synchrony)이 (a)우울 판별하나 (b)MDD<HC 기전 (c)★face+audio 각각을 *넘어서*
 정보 주나(=모달이 아니라 협응이 핵심=novel). D-Vlog(두 모달 신호有)+LMVD.
결과: results/exp74_crossmodal.csv
"""
import numpy as np, warnings, csv, glob
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
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10; N=300; K=20
def resamp(x,n=N):
    x=np.asarray(x,float)
    if len(x)<3: return np.zeros(n)
    return np.interp(np.linspace(0,len(x)-1,n),np.arange(len(x)),x)
def smooth(x,w=5):
    if len(x)<w:return x
    return np.convolve(x,np.ones(w)/w,mode='same')
def coord_feats(face_env, voice_env):
    fe=resamp(smooth(face_env)); ve=resamp(smooth(voice_env))
    fe=(fe-fe.mean())/(fe.std()+1e-6); ve=(ve-ve.mean())/(ve.std()+1e-6)
    r0=np.corrcoef(fe,ve)[0,1]
    # cross-corr over ±30
    best=r0;lag=0
    for L in range(-30,31):
        if L<0: a,b=fe[:L],ve[-L:]
        elif L>0: a,b=fe[L:],ve[:-L]
        else: a,b=fe,ve
        if len(a)>10:
            c=np.corrcoef(a,b)[0,1]
            if abs(c)>abs(best): best=c;lag=L
    # windowed 협응 일관성
    W=N//6; wc=[np.corrcoef(fe[i:i+W],ve[i:i+W])[0,1] for i in range(0,N-W,W)]
    wc=[c for c in wc if not np.isnan(c)]
    return np.nan_to_num(np.array([r0,best,lag/30.0,np.mean(wc) if wc else 0,np.std(wc) if wc else 0]))
def cov_pca(X,pca):
    Z=pca.transform(X);c,_=ledoit_wolf(Z);return c+1e-3*np.eye(Z.shape[1])
def cv(builders, y):
    y=np.array(y);res={}
    for name,build in builders.items():
        a=[]
        for s in range(SEEDS):
            skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
            for tr,te in skf.split(np.zeros(len(y)),y):
                X=np.nan_to_num(build(tr))
                sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
                pb[te]=cl.decision_function(sc.transform(X[te]))
            a.append(roc_auc_score(y,pb))
        res[name]=(np.mean(a),np.std(a))
    return res
def analyze(FAC,AUD,face_env,voice_env,y,name):
    y=np.array(y)
    CO=np.array([coord_feats(face_env[i],voice_env[i]) for i in range(len(y))])
    fp=PCA(min(K,FAC[0].shape[1])).fit(np.vstack([f[::3] for f in FAC]))
    ap=PCA(min(K,AUD[0].shape[1])).fit(np.vstack([a[::3] for a in AUD]))
    FCOV=[cov_pca(f,fp) for f in FAC]; ACOV=[cov_pca(a,ap) for a in AUD]
    def Tface(tr): return TangentSpace(metric='riemann').fit(np.array(FCOV)[tr]).transform(np.array(FCOV))
    def Taud(tr): return TangentSpace(metric='riemann').fit(np.array(ACOV)[tr]).transform(np.array(ACOV))
    builders={
      'face':lambda tr:Tface(tr),
      'audio':lambda tr:Taud(tr),
      'face+audio':lambda tr:np.column_stack([Tface(tr),Taud(tr)]),
      'coord만':lambda tr:CO,
      'face+audio+coord':lambda tr:np.column_stack([Tface(tr),Taud(tr),CO]),
    }
    res=cv(builders,y)
    # 기전
    r0=CO[:,0]; md,hc=r0[y==1].mean(),r0[y==0].mean();_,p=mannwhitneyu(r0[y==1],r0[y==0])
    print(f'\n===== {name} (n={len(y)}, 우울{int(y.sum())}) =====',flush=True)
    print(f'  [기전] 얼굴-음성 협응(corr) MDD={md:.3f} HC={hc:.3f} ({"↓우울(decouple)" if md<hc else "↑"}) p={p:.4f}',flush=True)
    for k in ['face','audio','face+audio','coord만','face+audio+coord']:
        print(f'  {k:18s} AUC={res[k][0]:.3f}±{res[k][1]:.3f}',flush=True)
    add=res['face+audio+coord'][0]-res['face+audio'][0]
    print(f'  → ★협응이 face+audio 넘어 추가하는가: Δ={add:+.3f} {"★novel(협응 자체가 정보)" if add>0.015 else "추가 미미"}',flush=True)
    return [name,len(y),md,hc,p,res['face'][0],res['audio'][0],res['face+audio'][0],res['coord만'][0],res['face+audio+coord'][0]]
# ---- D-Vlog ----
def load_dvlog():
    R=B/'D-Vlog'
    def nf(V):
        T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
        sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
    FAC=[];AUD=[];fenv=[];venv=[];y=[]
    for r in csv.DictReader(open(R/'labels.csv')):
        idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
        if not(fv.exists() and fa.exists()):continue
        try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
        except:continue
        if V.ndim!=2 or V.shape[0]<60 or V.shape[1]!=136 or Au.ndim!=2 or len(Au)<10:continue
        Vn=nf(V)
        FAC.append(Vn);AUD.append(Au)
        fenv.append(np.linalg.norm(np.diff(Vn,axis=0),axis=1))  # 얼굴 활동
        venv.append(np.abs(Au).mean(1))  # 음성 활동
        y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
    return FAC,AUD,fenv,venv,y
# ---- LMVD ----
def load_lmvd():
    V=B/'LMVD/extracted/Video_feature'; Au=B/'LMVD/extracted/Audio_feature'
    AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
    def lab(i):
        if (1<=i<=601) or (1117<=i<=1423): return 1
        if (602<=i<=1116) or (1425<=i<=1824): return 0
        return None
    FAC=[];AUD=[];fenv=[];venv=[];y=[]
    for f in sorted(glob.glob(str(V/'*.csv'))):
        sid=Path(f).stem;l=lab(int(sid))
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
        fa=np.nan_to_num(np.array(fe))
        FAC.append(fa);AUD.append(au)
        fenv.append(np.linalg.norm(np.diff((fa-fa.mean(0))/(fa.std(0)+1e-6),axis=0),axis=1))
        venv.append(np.abs(au).mean(1));y.append(l)
    return FAC,AUD,fenv,venv,y
out=[]
print('D-Vlog 로딩...',flush=True); a=load_dvlog(); out.append(analyze(*a,'D-Vlog'))
print('LMVD 로딩...',flush=True); a=load_lmvd(); out.append(analyze(*a,'LMVD'))
print('\n판정: 협응 MDD<HC 유의 + face+audio+coord가 face+audio 넘음(+0.015) → cross-modal 협응 붕괴 = novel 마커.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp74_crossmodal.csv','w') as f_:
    f_.write('corpus,n,coord_MDD,coord_HC,coord_p,face,audio,face_audio,coord_only,all\n')
    for r in out:f_.write(','.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in r)+'\n')
print('DONE → exp74_crossmodal.csv',flush=True)
