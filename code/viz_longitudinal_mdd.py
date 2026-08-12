"""
viz_longitudinal — 인터뷰 내 표정 변화가 가장 큰 피험자의 얼굴을 시간순 6장으로 landmark 재구성.
"진짜 종단(여러 방문)은 향후, 여기선 한 인터뷰 내 시간축 변화 illustration".
CMDC 사용(landmark x_/y_ in-file). 변화점수 = AU_r 시간 std 합. 정렬·스케일 정규화.
저장: results/longitudinal_face_MDD.png
"""
import numpy as np, warnings, openpyxl
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
C=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/CMDC/extracted')
NAVY='#1f3b6f'; RED='#C00000'
PARTS={'jaw':range(0,17),'r_brow':range(17,22),'l_brow':range(22,27),'nose':range(27,36),
       'r_eye':range(36,42),'l_eye':range(42,48),'mouth_out':range(48,60),'mouth_in':range(60,68)}
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx'); ws=wb.active
rows=list(ws.iter_rows(values_only=True)); hd=list(rows[0]); iID,iMDD=hd.index('ID'),hd.index('MDD')
CL={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}

def load(subj,q):
    f=C/subj/f'Q{q}.csv'
    if not f.exists(): return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:
        xi=[h.index(f'x_{i}') for i in range(68)]; yi=[h.index(f'y_{i}') for i in range(68)]
        oi=h.index('success'); ci=h.index('confidence'); ai=[h.index(c) for c in AU]
    except: return None
    X,Y,A=[],[],[]
    with open(f) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1 or float(v[ci])<0.85: continue
                X.append([float(v[i]) for i in xi]); Y.append([float(v[i]) for i in yi]); A.append([float(v[i]) for i in ai])
            except: pass
    if len(X)<30: return None
    return np.array(X),np.array(Y),np.array(A)

# 변화점수 = AU 시간 std 합 (표정 움직임 큰 사람). 가장 긴 질문클립 사용.
best=None
scores=[]
for s,l in CL.items():
    cand=[]
    for q in range(1,13):
        d=load(s,q)
        if d is not None: cand.append((len(d[0]),q,d))
    if not cand: continue
    cand.sort(reverse=True); n,q,(X,Y,A)=cand[0]  # 가장 긴 클립
    score=A.std(0).sum()
    scores.append((score,s,l,q,X,Y,A))
scores.sort(reverse=True,key=lambda r:r[0])
print('표정변화 상위 5명 (AU 시간std 합):',flush=True)
for sc,s,l,q,X,Y,A in scores[:5]:
    print(f'  {s} {"MDD" if l else "HC"} Q{q} n={len(X)} score={sc:.1f}',flush=True)
# force MDD subject for depression-story version
try:
    sc,subj,lab,q,X,Y,A=[r for r in scores if r[1]=="MDD22"][0]
except:
    sc,subj,lab,q,X,Y,A=scores[0]

# 6 시점 윈도우 평균 landmark
T=6; L=len(X); idx=np.linspace(0,L,T+1).astype(int)
def align(x,y):
    # y축 뒤집기(이미지좌표), 눈사이 거리로 스케일, 코끝(30) 중심
    y=-y
    le=np.array([x[36:42].mean(),y[36:42].mean()]); re=np.array([x[42:48].mean(),y[42:48].mean()])
    d=np.linalg.norm(le-re)+1e-6
    cx,cy=x[30],y[30]
    return (x-cx)/d,(y-cy)/d
fig,axes=plt.subplots(1,T,figsize=(2.1*T,3.0))
for k in range(T):
    a,b=idx[k],idx[k+1]
    xm=X[a:b].mean(0); ym=Y[a:b].mean(0)
    xx,yy=align(xm,ym)
    ax=axes[k]
    for name,ii in PARTS.items():
        ii=list(ii); ax.plot(xx[ii],yy[ii],c=NAVY,lw=1.4)
    ax.scatter(xx,yy,c=RED,s=8,zorder=3)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f't={int(100*k/(T-1))}%',fontsize=10)
grp='MDD (depressed)' if lab else 'HC (healthy)'
fig.suptitle(f'Within-session facial change over time — subject {subj} [{grp}]\n(illustration; true longitudinal tracking = future work)',fontsize=11)
plt.tight_layout(rect=[0,0,1,0.90])
plt.savefig('/home/hyuneun/disk_b/🟡facial-prodrome/results/longitudinal_face_MDD.png',dpi=140); plt.close()
print(f'\nDONE → longitudinal_face.png (subject {subj}, {grp}, Q{q}, {L} frames)',flush=True)
