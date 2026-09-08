"""
exp59 — idea E(문헌근거): 다채널 behavioral coupling = AU + head pose + gaze.
선점작(arXiv2407.13753)·내 기존작 = 전부 inter-AU only. PDF1/PDF4는 head pose·gaze(head turning,
slow eye movement, gaze aversion)가 핵심이라 함. → AU와 head/gaze의 '채널간 coordination'은 미탐색.
질문: AU+head+gaze coupling이 AU-only coupling을 넘나? cross-block(AU×gaze, AU×head) 단독은?
채널: AU(코퍼스별) + gaze_angle_x,y + pose_Rx,Ry,Rz. 결과: results/exp59_multichannel.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=10
HG=['gaze_angle_x','gaze_angle_y','pose_Rx','pose_Ry','pose_Rz']  # 5 head/gaze 채널
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
def cov_t(M):  # M: (T,D) 다채널 → z-score → ledoit-wolf cov
    M=np.nan_to_num(np.asarray(M,float)); seg=(M-M.mean(0))/(M.std(0)+1e-6); seg=np.nan_to_num(seg)
    c,_=ledoit_wolf(seg); return c+1e-4*np.eye(M.shape[1])
def cv(COV,y):
    COV=np.array(COV);y=np.array(y);a=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(COV,y):
            X=TangentSpace(metric='riemann').fit(COV[tr]).transform(COV)
            sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(X[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(X[te]))
        a.append(roc_auc_score(y,pb))
    return np.mean(a),np.std(a)
def analyze(mats, y, nAU, name):
    """mats: per-subj (T, nAU+5). AU블록=[:nAU], head/gaze=[nAU:]"""
    y=np.array(y)
    if len(y)<10 or len(set(y))<2:
        print(f'\n===== {name}: 로드 {len(y)}명 — 스킵(데이터 부족) =====',flush=True); return None
    COV_au=[cov_t(M[:,:nAU]) for M in mats]
    COV_full=[cov_t(M) for M in mats]
    ra=cv(COV_au,y); rf=cv(COV_full,y)
    print(f'\n===== {name} (n={len(y)}, 우울{y.sum()}, 채널 AU{nAU}+head/gaze5) =====',flush=True)
    print(f'  AU-only coupling      AUC={ra[0]:.3f}±{ra[1]:.3f}',flush=True)
    print(f'  AU+head+gaze coupling AUC={rf[0]:.3f}±{rf[1]:.3f}  Δ{rf[0]-ra[0]:+.3f} {"★head/gaze 추가" if rf[0]>ra[0]+0.02 else "추가 미미"}',flush=True)
    # cross-block coordination 단독: AU×(head/gaze) 상관 벡터
    cross=[]
    for M in mats:
        seg=(M-M.mean(0))/(M.std(0)+1e-6); C=np.corrcoef(seg.T)
        cb=C[:nAU,nAU:].flatten()  # AU×head/gaze 상호상관
        cross.append(np.nan_to_num(cb))
    cross=np.array(cross); ac=[]
    for s in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(y))
        for tr,te in skf.split(cross,y):
            sc=StandardScaler().fit(cross[tr]);cl=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(cross[tr]),y[tr])
            pb[te]=cl.decision_function(sc.transform(cross[te]))
        ac.append(roc_auc_score(y,pb))
    print(f'  cross-block(AU×head/gaze 협응) 단독 AUC={np.mean(ac):.3f}±{np.std(ac):.3f}',flush=True)
    return [name,len(y),ra[0],rf[0],np.mean(ac)]
out=[]
# ---- CMDC ----
C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
def cq(s,q):
    f=C/s/f'Q{q}.csv'
    if not f.exists():return None
    h=[x.strip() for x in open(f).readline().split(',')]
    try:oi=h.index('success');ai=[h.index(c) for c in AUc]+[h.index(c) for c in HG]
    except:return None
    fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
    return np.array(fe) if fe else None
M=[];Y=[]
for s,l in cl.items():
    parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
    if parts and sum(len(p) for p in parts)>=60: M.append(np.vstack(parts));Y.append(l)
out.append(analyze(M,Y,len(AUc),'CMDC'))
# ---- DAIC (AU/gaze/pose 별도파일 병합) ----
D=B/'DAIC_WOZ';dl={}
for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
    p=D/f
    if p.exists():
        for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
def dload(pid):
    pa=D/f'{pid}_CLNF_AUs.txt'; pg=D/f'{pid}_CLNF_gaze.txt'; pp=D/f'{pid}_CLNF_pose.txt'
    if not(pa.exists() and pg.exists() and pp.exists()):return None
    def rd(p):
        h=[x.strip() for x in open(p).readline().split(',')]; return h,[ln.split(',') for ln in open(p).readlines()[1:]]
    ha,ra=rd(pa); hg,rg=rd(pg); hp,rp=rd(pp)
    try:
        oi=ha.index('success'); aidx=[ha.index(c) for c in AUd]
        gx0,gx1,gy0,gy1=hg.index('x_0'),hg.index('x_1'),hg.index('y_0'),hg.index('y_1')  # CLNF: eye gaze 벡터
        rx,ry,rz=hp.index('Rx'),hp.index('Ry'),hp.index('Rz')  # CLNF pose: 접두사 없음
    except:return None
    n=min(len(ra),len(rg),len(rp)); out=[]
    for i in range(n):
        va=ra[i]
        try:
            if int(float(va[oi]))!=1:continue
            gxv=(float(rg[i][gx0])+float(rg[i][gx1]))/2; gyv=(float(rg[i][gy0])+float(rg[i][gy1]))/2
            row=[float(va[j]) for j in aidx]+[gxv,gyv,float(rp[i][rx]),float(rp[i][ry]),float(rp[i][rz])]
            out.append(row)
        except:pass
    return np.array(out) if len(out)>=60 else None
M=[];Y=[]
for pid,l in dl.items():
    m=dload(pid)
    if m is not None: M.append(m);Y.append(l)
out.append(analyze(M,Y,len(AUd),'DAIC'))
# ---- E-DAIC ----
E=B/'E-DAIC';el={}
for f in ['train_split.csv','dev_split.csv','test_split.csv']:
    p=E/'labels'/f
    if p.exists():
        for r in csv.DictReader(open(p)):
            pid=r['Participant_ID'].strip();b=r.get('PHQ_Binary') or r.get('PHQ8_Binary')
            if b not in(None,''):el[pid]=int(float(b))
def eload(pid):
    fs=glob.glob(str(E/'extracted'/f'{pid}_P'/'features'/f'{pid}_OpenFace*.csv'))
    if not fs:return None
    h=[x.strip() for x in open(fs[0]).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]+[h.index(c) for c in HG]
    except:return None
    fe=[]
    for ln in open(fs[0]).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    return np.array(fe) if len(fe)>=60 else None
M=[];Y=[]
for pid,l in el.items():
    m=eload(pid)
    if m is not None: M.append(m);Y.append(l)
out.append(analyze(M,Y,len(AUc),'E-DAIC'))
# ---- LMVD ----
V=B/'LMVD/extracted/Video_feature'
def lab(i):
    if (1<=i<=601) or (1117<=i<=1423): return 1
    if (602<=i<=1116) or (1425<=i<=1824): return 0
    return None
M=[];Y=[]
for f in sorted(glob.glob(str(V/'*.csv'))):
    l=lab(int(Path(f).stem))
    if l is None:continue
    h=[x.strip() for x in open(f).readline().split(',')]
    try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AUc]+[h.index(c) for c in HG]
    except:continue
    fe=[]
    for ln in open(f).readlines()[1:]:
        v=ln.split(',')
        try:
            if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
            fe.append([float(v[i]) for i in ai])
        except:pass
    if len(fe)>=60: M.append(np.array(fe));Y.append(l)
out.append(analyze(M,Y,len(AUc),'LMVD'))
print('\n판정: AU+head+gaze가 AU-only 대비 ≥2코퍼스 +0.02↑ → 다채널 behavioral coupling=비선점 novelty 후보.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp59_multichannel.csv','w') as f:
    f.write('corpus,n,AU_only,AU_head_gaze,cross_block\n')
    for r in [x for x in out if x]:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp59_multichannel.csv',flush=True)
