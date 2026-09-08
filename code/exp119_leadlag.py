"""exp119 — 얼굴-음성 lead-lag (방향성/시차). 미탐색 각.
얼굴 활동량 vs 음성 활동량 교차상관을 시차(lag)별로. peak lag(어느쪽이 이끄나)·peak 강도 MDD vs HC.
우울=정신운동지연이면 lag 지연/약화 예상. D-Vlog(per-second, 최상 음성) 탐침.
"""
import numpy as np, warnings, csv
from pathlib import Path
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog');LAGS=range(-5,6)  # ±5초
def nf(V):
    T2=V.shape[0];P=V.reshape(T2,68,2).astype(float);P=P-P.mean(1,keepdims=True)
    sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T2,136)
def leadlag(face,voice):
    n=min(len(face),len(voice))
    if n<30:return None
    f=(face[:n]-face[:n].mean())/(face[:n].std()+1e-9);v=(voice[:n]-voice[:n].mean())/(voice[:n].std()+1e-9)
    xc={}
    for L in LAGS:
        if L<0:a,b=f[:L],v[-L:]          # voice가 앞섬(face lag)
        elif L>0:a,b=f[L:],v[:-L]        # face가 앞섬
        else:a,b=f,v
        if len(a)<20 or a.std()<1e-6 or b.std()<1e-6:xc[L]=np.nan;continue
        c=np.corrcoef(a,b)[0,1];xc[L]=c if np.isfinite(c) else np.nan
    L_arr=np.array(list(LAGS));c_arr=np.array([xc[L] for L in LAGS])
    if not np.isfinite(c_arr).any():return None
    ac=np.abs(c_arr)
    peak_lag=L_arr[np.nanargmax(ac)];peak=np.nanmax(ac)
    asym=np.nansum(c_arr[L_arr>0])-np.nansum(c_arr[L_arr<0])  # 양수면 face lead
    return peak_lag,peak,asym
Peak=[];PL=[];ASY=[];Y=[]
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
    if not(fv.exists() and fa.exists()):continue
    try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
    except:continue
    if V.ndim!=2 or V.shape[0]<40 or V.shape[1]!=136 or Au.ndim!=2:continue
    P=nf(V);face=np.concatenate([[0],np.linalg.norm(np.diff(P,axis=0),axis=1)]);voice=np.linalg.norm(Au,axis=1)
    res=leadlag(face,voice)
    if res is None:continue
    Peak.append(res[1]);PL.append(res[0]);ASY.append(res[2]);Y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
Peak=np.array(Peak);PL=np.array(PL);ASY=np.array(ASY);Y=np.array(Y)
print(f'n={len(Y)} MDD={Y.sum()}',flush=True)
for nm,a in [('peak 강도',Peak),('peak lag(초)',PL),('방향성(양수=face lead)',ASY)]:
    mm=a[Y==1];hh=a[Y==0];mm=mm[np.isfinite(mm)];hh=hh[np.isfinite(hh)]
    try:_,p=mannwhitneyu(mm,hh,alternative='two-sided')
    except:p=1
    print(f'  {nm:22s} MDD={mm.mean():+.3f} HC={hh.mean():+.3f} p={p:.2e}',flush=True)
# lag별 평균 곡선
print('--- 시차별 평균 교차상관 (MDD vs HC) ---',flush=True)
allxc={L:[] for L in LAGS}
for r in csv.DictReader(open(R/'labels.csv')):
    idx=r['index'].strip();fv=R/idx/f'{idx}_visual.npy';fa=R/idx/f'{idx}_acoustic.npy'
    if not(fv.exists() and fa.exists()):continue
    try:V=np.load(fv);Au=np.nan_to_num(np.load(fa).astype(float))
    except:continue
    if V.ndim!=2 or V.shape[0]<40 or V.shape[1]!=136 or Au.ndim!=2:continue
    P=nf(V);face=np.concatenate([[0],np.linalg.norm(np.diff(P,axis=0),axis=1)]);voice=np.linalg.norm(Au,axis=1)
    n=min(len(face),len(voice))
    if n<30:continue
    f=(face[:n]-face[:n].mean())/(face[:n].std()+1e-9);v=(voice[:n]-voice[:n].mean())/(voice[:n].std()+1e-9)
    lab=1 if r['label'].strip().lower().startswith('depress') else 0
    for L in LAGS:
        if L<0:a,b=f[:L],v[-L:]
        elif L>0:a,b=f[L:],v[:-L]
        else:a,b=f,v
        if len(a)>=20 and a.std()>1e-6 and b.std()>1e-6:
            c=np.corrcoef(a,b)[0,1]
            if np.isfinite(c):allxc[L].append((lab,c))
for L in LAGS:
    d=allxc[L];m=[c for l,c in d if l==1];h=[c for l,c in d if l==0]
    print(f'  lag{L:+d}s: MDD={np.mean(m):+.3f} HC={np.mean(h):+.3f}',flush=True)
print('DONE',flush=True)
