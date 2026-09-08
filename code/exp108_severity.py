"""exp108 — severity 확장. 협응+활동량 feature로 PHQ 심각도 회귀(CMDC PHQtotal, E-DAIC PHQ_Score).
Ridge 5-fold CV. CCC/MAE/Pearson r. 이진 탐지→연속 심각도로 태스크 확장.
"""
import numpy as np, warnings, csv, os, re, glob
import pandas as pd
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/'
def logm_vech(C):
    ev,U=np.linalg.eigh(C);ev=np.clip(ev,1e-6,None);L=U@np.diag(np.log(ev))@U.T
    iu=np.triu_indices(len(C));return L[iu[0],iu[1]]
def feat(series):
    Z=(series-series.mean(0))/(series.std(0)+1e-6)
    C,_=ledoit_wolf(Z);coord=logm_vech(C+1e-3*np.eye(series.shape[1]))
    return np.concatenate([coord,series.mean(0)])  # 협응 + 활동량(mean)
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
def regress(X,y,name):
    X=np.array(X);y=np.array(y,float);kf=KFold(5,shuffle=True,random_state=0)
    pred=np.zeros(len(y))
    for tr,te in kf.split(X):
        sc=StandardScaler().fit(X[tr])
        m=RidgeCV(alphas=[1,10,50,200,1000,5000]).fit(sc.transform(X[tr]),y[tr])  # train내부 GCV로 alpha
        pred[te]=m.predict(sc.transform(X[te]))
    c=ccc(y,pred);mae=np.abs(y-pred).mean();r=pearsonr(y,pred)[0]
    print(f'[{name:7s} n={len(y)}] CCC={c:.3f}  MAE={mae:.2f}  r={r:.3f}  (PHQ 범위 {y.min():.0f}~{y.max():.0f})',flush=True)
    RES[name]=(y,pred,c,r)
    return c,mae,r
RES={}
# CMDC
info=pd.read_excel(DATA+'CMDC/extracted/SubjectInfo.xlsx');info['ID']=info['ID'].astype(str).str.strip()
phq={r['ID']:r['PHQtotal'] for _,r in info.iterrows() if not pd.isna(r['PHQtotal'])}
X=[];y=[]
for dfold in sorted(glob.glob(DATA+'CMDC/extracted/*/')):
    name=os.path.basename(dfold.rstrip('/'))
    if name not in phq:continue
    segs=[]
    for f in glob.glob(dfold+'Q*.csv'):
        if not re.match(r'^Q\d+\.csv$',os.path.basename(f)):continue
        ps=persec(f)
        if ps is not None:segs.append(ps)
    if not segs:continue
    s=np.vstack(segs)
    if len(s)<40:continue
    X.append(feat(s));y.append(float(phq[name]))
regress(X,y,'CMDC')
# E-DAIC
lab={}
for spn in ['train','dev','test']:
    p=DATA+f'E-DAIC/labels/{spn}_split.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            pid=r.get('Participant_ID','').strip();sc=r.get('PHQ_Score','').strip()
            if pid and sc not in('',None):
                try:lab[pid]=float(sc)
                except:pass
X=[];y=[]
for dfold in sorted(glob.glob(DATA+'E-DAIC/extracted/*_P/')):
    pid=os.path.basename(dfold.rstrip('/')).replace('_P','')
    if pid not in lab:continue
    f=dfold+f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'
    if not os.path.exists(f):continue
    ps=persec(f)
    if ps is None or len(ps)<40:continue
    X.append(feat(ps));y.append(lab[pid])
regress(X,y,'E-DAIC')
print('\n[참고] 지난 severity(초기)=CMDC CCC~0.38. 개선/유지 확인. 심각도는 이진탐지의 보조 확장.',flush=True)
# fig11 산점도
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for c2 in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    if os.path.exists(c2):fm.fontManager.addfont(c2);plt.rcParams['font.family']=fm.FontProperties(fname=c2).get_name()
plt.rcParams['axes.unicode_minus']=False
order=[k for k in ['CMDC','E-DAIC'] if k in RES]
fig,axs=plt.subplots(1,len(order),figsize=(5.4*len(order),4.6))
if len(order)==1:axs=[axs]
for ax,k in zip(axs,order):
    y,p,c,r=RES[k];col='#1565C0' if c>0.3 else '#9E9E9E'
    ax.scatter(y,p,c=col,alpha=0.6,s=30)
    lo=min(y.min(),p.min());hi=max(y.max(),p.max());ax.plot([lo,hi],[lo,hi],'--',color='#c0392b',lw=1)
    ax.set_xlabel('실제 PHQ 점수');ax.set_ylabel('예측 PHQ 점수')
    ax.set_title(f'{k}  (CCC={c:.2f}, r={r:.2f})',fontsize=12)
fig.suptitle('우울 심각도(PHQ) 예측 — 협응+활동량 (심각도 확장)',fontsize=13)
plt.tight_layout();plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/figs/fig11_severity.png',dpi=145,bbox_inches='tight')
print('DONE fig11_severity.png',flush=True)
