"""exp92 — hypoexpressivity p값 검증. 피험자별 표정 활동량 Mann-Whitney (LMVD AU, D-Vlog landmark)."""
import numpy as np, glob, os, re, warnings
import pandas as pd
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
B='/home/hyuneun/disk_b/🟡facial-prodrome/data/'
# --- LMVD (AU intensity 평균 = 활동량) ---
def lmvd_label(i):
    return 1 if (1<=i<=601 or 1117<=i<=1423) else 0  # dep=1 / normal=0
AUcols=None; dep=[]; nor=[]
vids=sorted(glob.glob(B+'LMVD/Video_feature/*.csv'))
for f in vids:
    m=re.search(r'(\d+)',os.path.basename(f))
    if not m: continue
    i=int(m.group(1))
    try: df=pd.read_csv(f)
    except: continue
    df.columns=[c.strip() for c in df.columns]
    au=[c for c in df.columns if re.match(r'AU\d+_r$',c)]
    if not au: continue
    val=df[au].values.astype(float)
    if len(val)<10: continue
    expr=np.nanmean(val)  # 전체 AU 평균 활동량
    (dep if lmvd_label(i)==1 else nor).append(expr)
if dep and nor:
    u,p=mannwhitneyu(dep,nor,alternative='two-sided')
    print(f'LMVD 활동량  MDD={np.mean(dep):.3f}(n{len(dep)})  HC={np.mean(nor):.3f}(n{len(nor)})  p={p:.2e}')
else:
    print('LMVD: 데이터 부족')
# --- D-Vlog (landmark 프레임간 이동량 평균 = 활동량) ---
import csv
R=B+'D-Vlog/'
rows=list(csv.DictReader(open(R+'labels.csv')))
dd=[];hh=[]
for r in rows:
    idx=r['index'].strip()
    fv=f'{R}{idx}/{idx}_visual.npy'
    if not os.path.exists(fv): continue
    try: V=np.load(fv)
    except: continue
    if V.ndim!=2 or V.shape[0]<10 or V.shape[1]!=136: continue
    P=V.reshape(V.shape[0],68,2).astype(float)
    mov=np.linalg.norm(np.diff(P,axis=0),axis=2).mean()  # 프레임간 평균 이동량
    lab=1 if r['label'].strip().lower().startswith('depress') else 0
    (dd if lab==1 else hh).append(mov)
if dd and hh:
    u,p=mannwhitneyu(dd,hh,alternative='two-sided')
    print(f'D-Vlog 활동량  MDD={np.mean(dd):.4f}(n{len(dd)})  HC={np.mean(hh):.4f}(n{len(hh)})  p={p:.2e}')
else:
    print('D-Vlog: 데이터 부족')
print('DONE')
