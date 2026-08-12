"""
viz_face — CMDC 랜드마크(68점) 시각화. 점+선(부위별 색). 임시 확인용.
우울군 vs 정상군 평균 얼굴 비교.
"""
import numpy as np, warnings, openpyxl
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
# 68 landmark 부위 (dlib 표준)
PARTS={'jaw':range(0,17),'r_brow':range(17,22),'l_brow':range(22,27),
       'nose':range(27,36),'r_eye':range(36,42),'l_eye':range(42,48),
       'mouth_out':range(48,60),'mouth_in':range(60,68)}
COL={'jaw':'gray','r_brow':'C1','l_brow':'C0','nose':'purple','r_eye':'red','l_eye':'green','mouth_out':'brown','mouth_in':'brown'}
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0]); iID,iMDD=hd.index('ID'),hd.index('MDD')
CL={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def landmarks_mean(subj,q=3):
    f=C/subj/f'Q{q}.csv'
    if not f.exists(): return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:
        xi=[h.index(f'x_{i}') for i in range(68)]; yi=[h.index(f'y_{i}') for i in range(68)]; oi=h.index('success')
    except: return None
    xs=[];ys=[]
    with open(f) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                xs.append([float(v[i]) for i in xi]); ys.append([float(v[i]) for i in yi])
            except: pass
    if len(xs)<5: return None
    return np.array(xs).mean(0), np.array(ys).mean(0)
def draw(ax,x,y,title):
    ax.scatter(x,y,c='blue',s=20,zorder=3)
    for name,idx in PARTS.items():
        idx=list(idx); ax.plot(x[idx],y[idx],c=COL[name],lw=1.5,label=name if name in('jaw',) else None)
    ax.set_title(title); ax.invert_yaxis(); ax.set_aspect('equal'); ax.axis('off')
# 우울/정상 대표 각 1명 + 평균
dep=[s for s in CL if CL[s]==1]; hc=[s for s in CL if CL[s]==0]
fig,axes=plt.subplots(1,2,figsize=(10,5))
for s in dep:
    r=landmarks_mean(s)
    if r: draw(axes[0],r[0],r[1],f'MDD (depressed) — {s}, Q3(sleep)'); break
for s in hc:
    r=landmarks_mean(s)
    if r: draw(axes[1],r[0],r[1],f'HC (control) — {s}, Q3(sleep)'); break
plt.suptitle('CMDC OpenFace 68 landmarks (mean over Q3 sleep segment)')
plt.tight_layout(); plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/results/viz_face_sample.png',dpi=120)
print('saved viz_face_sample.png')
