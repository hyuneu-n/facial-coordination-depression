"""exp120 — lead-lag 재현 (LMVD·E-DAIC). 초단위 정렬 후 얼굴-음성 교차상관 peak lag·강도 MDD vs HC.
D-Vlog: MDD peak lag +0.46s(지연)·강도 약. 재현되면 novelty.
"""
import numpy as np, warnings, os, re, glob, csv
import pandas as pd
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/';LAGS=range(-5,6)
def persec_series(vals,times):
    # vals:(T,) 값, times:(T,) 초 → 초단위 평균
    sec=np.floor(times).astype(int);sec-=sec.min()
    n=sec.max()+1;out=np.zeros(n);cnt=np.zeros(n)
    for v,s in zip(vals,sec):out[s]+=v;cnt[s]+=1
    cnt[cnt==0]=1;return out/cnt
def leadlag(face,voice):
    n=min(len(face),len(voice))
    if n<30:return None
    f=(face[:n]-face[:n].mean())/(face[:n].std()+1e-9);v=(voice[:n]-voice[:n].mean())/(voice[:n].std()+1e-9)
    if f.std()<1e-6 or v.std()<1e-6:return None
    c_arr=[]
    for L in LAGS:
        if L<0:a,b=f[:L],v[-L:]
        elif L>0:a,b=f[L:],v[:-L]
        else:a,b=f,v
        if len(a)<20 or a.std()<1e-6 or b.std()<1e-6:c_arr.append(np.nan);continue
        c=np.corrcoef(a,b)[0,1];c_arr.append(c if np.isfinite(c) else np.nan)
    c_arr=np.array(c_arr);L_arr=np.array(list(LAGS))
    if not np.isfinite(c_arr).any():return None
    return L_arr[np.nanargmax(np.abs(c_arr))],np.nanmax(np.abs(c_arr))
def report(name,PL,Peak,Y):
    PL=np.array(PL);Peak=np.array(Peak);Y=np.array(Y)
    for nm,a in [('peak lag(초)',PL),('peak 강도',Peak)]:
        mm=a[Y==1];hh=a[Y==0];mm=mm[np.isfinite(mm)];hh=hh[np.isfinite(hh)]
        try:_,p=mannwhitneyu(mm,hh,alternative='two-sided')
        except:p=1
        print(f'[{name}] {nm:12s} MDD={mm.mean():+.3f} HC={hh.mean():+.3f} p={p:.2e}',flush=True)
# LMVD
def lm(i):return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
aud={}
for f in glob.glob(DATA+'LMVD/extracted/Audio_feature/*.npy'):
    m=re.search(r'(\d+)',os.path.basename(f))
    if m:aud[int(m.group(1))]=f
PL=[];Peak=[];Y=[]
for vf in sorted(glob.glob(DATA+'LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(vf))
    if not m:continue
    i=int(m.group(1))
    if i not in aud:continue
    try:dv=pd.read_csv(vf,usecols=lambda c:c.strip() in set(['timestamp','success']+AUS));dv.columns=[c.strip() for c in dv.columns]
    except:continue
    if 'timestamp' not in dv or not set(AUS).issubset(dv.columns):continue
    if 'success' in dv:dv=dv[dv['success']==1]
    if len(dv)<40:continue
    AU=dv[AUS].values.astype(float);ts=dv['timestamp'].values.astype(float)
    fmov=np.concatenate([[0],np.linalg.norm(np.diff(AU,axis=0),axis=1)])
    face=persec_series(fmov,ts)
    try:V=np.load(aud[i])
    except:continue
    dur=ts.max()-ts.min()
    ax=np.linspace(0,dur,len(V));voice=persec_series(np.linalg.norm(V,axis=1),ax)
    res=leadlag(face,voice)
    if res is None:continue
    PL.append(res[0]);Peak.append(res[1]);Y.append(lm(i))
report('LMVD ',PL,Peak,Y)
# E-DAIC
lab={}
for spn in ['train','dev','test']:
    p=DATA+f'E-DAIC/labels/{spn}_split.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            pid=r.get('Participant_ID','').strip();b=r.get('PHQ_Binary','').strip()
            if pid and b in('0','1'):lab[pid]=int(b)
PL=[];Peak=[];Y=[]
for dfold in sorted(glob.glob(DATA+'E-DAIC/extracted/*_P/')):
    pid=os.path.basename(dfold.rstrip('/')).replace('_P','')
    if pid not in lab:continue
    vf=dfold+f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv';af=dfold+f'features/{pid}_OpenSMILE2.3.0_egemaps.csv'
    if not(os.path.exists(vf) and os.path.exists(af)):continue
    try:
        dv=pd.read_csv(vf,usecols=lambda c:c.strip() in set(['timestamp','success']+AUS));dv.columns=[c.strip() for c in dv.columns]
        if 'success' in dv:dv=dv[dv['success']==1]
        AU=dv[AUS].values.astype(float);ts=dv['timestamp'].values.astype(float)
        da=pd.read_csv(af,sep=';');da.columns=[c.strip() for c in da.columns]
        loud=da['Loudness_sma3'].values.astype(float);lt=da['frameTime'].values.astype(float)
    except:continue
    if len(AU)<40 or len(loud)<40:continue
    fmov=np.concatenate([[0],np.linalg.norm(np.diff(AU,axis=0),axis=1)])
    face=persec_series(fmov,ts);voice=persec_series(loud,lt)
    res=leadlag(face,voice)
    if res is None:continue
    PL.append(res[0]);Peak.append(res[1]);Y.append(lab[pid])
report('E-DAIC',PL,Peak,Y)
print('\n[D-Vlog] peak lag MDD+0.46 HC+0.01 (p.023), 강도 MDD약(p.007). 재현 판정.',flush=True)
print('DONE',flush=True)
