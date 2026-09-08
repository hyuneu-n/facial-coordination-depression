"""exp111 — 조기+심각도 결합. CMDC PHQ를 세션 앞 X%만으로 예측되나? CCC vs fraction.
조기탐지(앞40%충분)가 심각도에도 성립하면 결합 발견.
"""
import numpy as np, warnings, os, re, glob
import pandas as pd
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/';FRACS=[0.2,0.4,0.6,0.8,1.0]
def logm_vech(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None);L=U@np.diag(np.log(ev))@U.T
    iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def feat(series):
    Z=(series-series.mean(0))/(series.std(0)+1e-6);C,_=ledoit_wolf(Z)
    return np.concatenate([logm_vech(C+1e-3*np.eye(series.shape[1])),series.mean(0)])
def persec(f):
    try:df=pd.read_csv(f,usecols=lambda c:c.strip() in set(['timestamp','success','confidence']+AUS))
    except:return None
    df.columns=[c.strip() for c in df.columns]
    if 'timestamp' not in df or not set(AUS).issubset(df.columns):return None
    if 'success' in df:df=df[df['success']==1]
    if len(df)<50:return None
    df=df.copy();df['sec']=np.floor(df['timestamp']).astype(int)
    return df.groupby('sec')[AUS].mean().values
def ccc(x,y):
    x=np.array(x);y=np.array(y);vx=x.var();vy=y.var();cov=((x-x.mean())*(y-y.mean())).mean()
    return 2*cov/(vx+vy+(x.mean()-y.mean())**2+1e-9)
# CMDC 로드 (per-second concat series + PHQ)
info=pd.read_excel(DATA+'CMDC/extracted/SubjectInfo.xlsx');info['ID']=info['ID'].astype(str).str.strip()
phq={r['ID']:r['PHQtotal'] for _,r in info.iterrows() if not pd.isna(r['PHQtotal'])}
SER=[];Y=[]
for dfold in sorted(glob.glob(DATA+'CMDC/extracted/*/')):
    name=os.path.basename(dfold.rstrip('/'))
    if name not in phq:continue
    segs=[persec(f) for f in glob.glob(dfold+'Q*.csv') if re.match(r'^Q\d+\.csv$',os.path.basename(f))]
    segs=[s for s in segs if s is not None]
    if not segs:continue
    s=np.vstack(segs)
    if len(s)<50:continue
    SER.append(s);Y.append(float(phq[name]))
Y=np.array(Y);print(f'CMDC n={len(Y)}, PHQ {Y.min():.0f}~{Y.max():.0f}',flush=True)
def regress_frac(frac):
    X=np.array([feat(s[:max(40,int(len(s)*frac))]) for s in SER])
    kf=KFold(5,shuffle=True,random_state=0);pred=np.zeros(len(Y))
    for tr,te in kf.split(X):
        sc=StandardScaler().fit(X[tr]);m=RidgeCV(alphas=[1,10,50,200,1000,5000]).fit(sc.transform(X[tr]),Y[tr])
        pred[te]=m.predict(sc.transform(X[te]))
    return ccc(Y,pred),pearsonr(Y,pred)[0]
print('=== CMDC 심각도(PHQ) 조기 예측 ===',flush=True)
ccs=[];rs=[]
for f in FRACS:
    c,r=regress_frac(f);ccs.append(c);rs.append(r)
    print(f'  앞 {int(f*100):3d}%: CCC={c:.3f}  r={r:.3f}',flush=True)
# fig13
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for c2 in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c2):fm.fontManager.addfont(c2);plt.rcParams['font.family']=fm.FontProperties(fname=c2).get_name()
plt.rcParams['axes.unicode_minus']=False
x=[f*100 for f in FRACS]
plt.figure(figsize=(6.2,4.4))
plt.plot(x,ccs,'-o',color='#1565C0',lw=2.2,label='CCC')
plt.plot(x,rs,'--s',color='#2E7D32',lw=1.8,label='Pearson r')
for xi,c in zip(x,ccs):plt.text(xi,c+0.02,f'{c:.2f}',ha='center',fontsize=9,fontweight='bold')
plt.xlabel('관찰한 세션 앞부분 (%)');plt.ylabel('심각도 예측 성능')
plt.title('심각도(PHQ)도 세션 앞부분으로 예측되나 — CMDC\n(조기탐지 × 심각도 결합)',fontsize=12)
plt.ylim(0,0.7);plt.legend(fontsize=10);plt.tight_layout()
plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig13_early_severity.png',dpi=145,bbox_inches='tight')
print('DONE fig13',flush=True)
