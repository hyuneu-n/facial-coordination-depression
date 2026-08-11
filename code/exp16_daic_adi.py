"""
exp16 — DAIC ADI(Anchor Diagnosticity Index) = 질문별 표정-우울 진단력.
실험1(재현·CI): 각 정형질문 앵커별 AUC + bootstrap 95% CI (10 seed).
어느 질문(앵커)이 진단적인가 = ADI 랭킹. CMDC와 비교할 교차언어 기반.
"""
import numpy as np, csv, warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
D=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
POST=8.0; SEEDS=list(range(10))
# 감정/증상 유도 질문 앵커 후보 (태그로 매칭). PHQ 증상축 라벨 병기.
ANCHORS={
 'happy_lasttime':'긍정(행복)','feel_lately':'기분','depression_diagnosed':'우울진단',
 'regret':'후회','control_temper':'분노','last_argument':'갈등','easy_sleep':'수면',
 'dream_job':'긍정(꿈)','memory_erase':'지우고픈기억','hard_decision':'스트레스',
 'feelguilty':'죄책감','feelbadly':'부정정서','sleep_affects':'수면영향','best_friend_describe':'관계',
}
def labels():
    lab={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)): lab[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
    return lab
def transcript(pid):
    p=D/f'{pid}_TRANSCRIPT.csv'; rows=[]
    if p.exists():
        for r in csv.DictReader(open(p),delimiter='\t'):
            try: rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '')))
            except: pass
    return rows
def au_series(pid):
    p=D/f'{pid}_CLNF_AUs.txt'
    if not p.exists(): return None,None
    h=[x.strip() for x in open(p).readline().split(',')]
    ti,oi=h.index('timestamp'),h.index('success'); ai=[h.index(c) for c in AU]
    ts,fe=[],[]
    with open(p) as fp:
        fp.readline()
        for ln in fp:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1: continue
                ts.append(float(v[ti])); fe.append([float(v[i]) for i in ai])
            except: pass
    return np.array(ts),np.array(fe)
def anchor_times(rows,tag):
    # Ellie 발화 value가 "tag (...)" 형식 → tag로 시작하는 것
    return [sp for st,sp,spk,val in rows if spk=='Ellie' and val.strip().lower().startswith(tag)]
lab=labels()
# 미리 로드
DATA={}
for pid in lab:
    ts,au=au_series(pid)
    if ts is not None: DATA[pid]=(ts,au,transcript(pid))
def feat(pid,tag):
    ts,au,rows=DATA[pid]
    base=au.mean(0); bs=au.std(0)+1e-6
    segs=[au[(ts>=a)&(ts<a+POST)] for a in anchor_times(rows,tag)]
    segs=[s for s in segs if len(s)>=6]
    if not segs: return None
    seg=(np.vstack(segs)-base)/bs
    return np.concatenate([seg.mean(0),seg.std(0)])
def adi(tag):
    F,y=[],[]
    for pid,l in lab.items():
        if pid not in DATA: continue
        f=feat(pid,tag)
        if f is not None: F.append(f); y.append(l)
    if len(y)<30 or sum(y)<8: return None
    F,y=np.array(F),np.array(y); aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); pb=np.zeros(len(y),float)
        for tr,te in skf.split(F,y):
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            clf.fit(F[tr],y[tr]); pb[te]=clf.decision_function(F[te])
        aucs.append(roc_auc_score(y,pb))
    aucs=np.array(aucs)
    return len(y),aucs.mean(),np.percentile(aucs,2.5),np.percentile(aucs,97.5)
print('=== DAIC ADI: 질문별 진단력 (10 seed, 95% CI) ===',flush=True)
print(f"{'질문태그':>20} {'증상축':>10} {'n':>4} {'AUC':>6} {'95%CI':>16}",flush=True)
print('-'*62,flush=True)
res=[]
for tag,axis in ANCHORS.items():
    r=adi(tag)
    if r is None: continue
    n,m,lo,hi=r; res.append((tag,axis,n,m,lo,hi))
for tag,axis,n,m,lo,hi in sorted(res,key=lambda x:-x[3]):
    print(f"{tag:>20} {axis:>10} {n:>4} {m:>6.3f} [{lo:.3f},{hi:.3f}]",flush=True)
print(f"\n[ADI 최고] {sorted(res,key=lambda x:-x[3])[0][0]} / [최저] {sorted(res,key=lambda x:x[3])[0][0]}",flush=True)
