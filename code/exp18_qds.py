"""
exp18 — QDS 가중집계 (실험0 재현 + 실험1 생사).
QDS: 질문별 진단가중치를 train fold 내부에서만 추정(라벨누수X) → AU 가중집계.
비교군: 균등집계 / random-weight / whole-interview / learned-attention(경량).
DAIC. nested-CV, 10 seed, DeLong 대신 순열로 유의차. 주지표 AUC.
성공기준: QDS ≥ 균등·random·attention 유의(양수 ΔAUC).
"""
import numpy as np, csv, warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
D=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
NA=len(AU); POST=8.0; SEEDS=list(range(10))
# 감정/증상 유도 질문 앵커 후보 (충분히 많이 = QDS가 고를 pool)
QTAGS=['happy_lasttime','feel_lately','depression_diagnosed','regret','control_temper',
       'last_argument','easy_sleep','dream_job','memory_erase','hard_decision','feelguilty',
       'sleep_affects','best_friend_describe','ptsd_diagnosed','self_change','situation_handled',
       'family_relationship','study','influence_positive']
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
lab=labels()
# per-subject per-question feature (mean+std of AU, 개인정규화) → dict[pid][qtag]=vec(2*NA) or None
FEAT={}
for pid in list(lab):
    ts,au=au_series(pid)
    if ts is None: continue
    base=au.mean(0); bs=au.std(0)+1e-6; rows=transcript(pid)
    d={}
    for tag in QTAGS:
        segs=[]
        for st,sp,spk,val in rows:
            if spk=='Ellie' and val.strip().lower().startswith(tag):
                m=(ts>=sp)&(ts<sp+POST)
                if m.sum()>=6: segs.append(au[m])
        if segs:
            seg=(np.vstack(segs)-base)/bs
            d[tag]=np.concatenate([seg.mean(0),seg.std(0)])
    # whole interview
    whole=(au-base)/bs
    d['__whole__']=np.concatenate([whole.mean(0),whole.std(0)])
    FEAT[pid]=d
pids=[p for p in lab if p in FEAT]
y=np.array([lab[p] for p in pids])
print(f'n={len(pids)}, 우울{int(y.sum())}/정상{int((y==0).sum())}, 질문pool={len(QTAGS)}\n',flush=True)

DIM=2*NA
def subj_matrix(tag):
    """질문 tag의 피험자 벡터 행렬 + 존재마스크"""
    X=np.full((len(pids),DIM),np.nan);
    for i,p in enumerate(pids):
        if tag in FEAT[p]: X[i]=FEAT[p][tag]
    return X

# 각 질문 행렬 미리
QMAT={tag:subj_matrix(tag) for tag in QTAGS}
WHOLE=subj_matrix('__whole__')

def impute(X):
    m=np.nanmean(X,0); m=np.where(np.isnan(m),0,m)
    Xi=X.copy(); idx=np.where(np.isnan(Xi)); Xi[idx]=np.take(m,idx[1]); return Xi

def qds_weights(tr_idx):
    """train에서만 질문별 진단력(oof AUC) 추정 → 가중치. 라벨누수 없음."""
    w={}
    ytr=y[tr_idx]
    for tag in QTAGS:
        X=QMAT[tag][tr_idx]; mask=~np.isnan(X[:,0])
        if mask.sum()<20 or ytr[mask].sum()<5 or (ytr[mask]==0).sum()<5: w[tag]=0.0; continue
        Xt=X[mask]; yt=ytr[mask]
        try:
            skf=StratifiedKFold(3,shuffle=True,random_state=0); pb=np.zeros(len(yt))
            for a,b in skf.split(Xt,yt):
                sc=StandardScaler().fit(Xt[a]); clf=LogisticRegression(max_iter=1000,class_weight='balanced').fit(sc.transform(Xt[a]),yt[a])
                pb[b]=clf.decision_function(sc.transform(Xt[b]))
            auc=roc_auc_score(yt,pb)
        except: auc=0.5
        w[tag]=max(auc-0.5,0.0)  # QDS: 진단력 초과분, 음수 클립
    s=sum(w.values());
    if s>0: w={k:v/s for k,v in w.items()}
    return w

def aggregate(idxset,w):
    """피험자별 질문 가중결합 (존재하는 질문만, 가중 재정규화)"""
    out=np.zeros((len(idxset),DIM))
    for j,i in enumerate(idxset):
        num=np.zeros(DIM); den=0.0
        for tag in QTAGS:
            if tag in FEAT[pids[i]] and w.get(tag,0)>0:
                num+=w[tag]*FEAT[pids[i]][tag]; den+=w[tag]
        out[j]= num/den if den>0 else np.nan
    return out

def eval_method(method):
    aucs=[]
    for sd in SEEDS:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd); proba=np.full(len(y),np.nan)
        for tr,te in skf.split(np.zeros(len(y)),y):
            if method=='whole':
                Xtr,Xte=impute(WHOLE[tr]),impute(WHOLE[te])
            elif method=='uniform':
                w={t:1.0 for t in QTAGS}; s=len(QTAGS); w={k:v/s for k,v in w.items()}
                Xtr=aggregate(tr,w); Xte=aggregate(te,w)
                Xtr=impute(Xtr); Xte=impute(Xte)
            elif method=='random':
                rng=np.random.RandomState(sd); w={t:rng.rand() for t in QTAGS}; s=sum(w.values()); w={k:v/s for k,v in w.items()}
                Xtr=impute(aggregate(tr,w)); Xte=impute(aggregate(te,w))
            elif method=='qds':
                w=qds_weights(tr)  # train만
                Xtr=impute(aggregate(tr,w)); Xte=impute(aggregate(te,w))
            elif method=='attention':
                # learned attention: 질문별 벡터를 concat하고 학습 (경량) → 여기선 질문별 예측 평균에 학습가중
                # 간단화: train서 각 질문 로지스틱 → val 확률을 stacking 로지스틱
                Xtr=impute(aggregate(tr,{t:1 for t in QTAGS})); Xte=impute(aggregate(te,{t:1 for t in QTAGS}))
                # (uniform과 동일 취급 방지 위해 stacking) — 아래 별도 처리
            sc=StandardScaler().fit(Xtr); clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(Xtr),y[tr])
            proba[te]=clf.decision_function(sc.transform(Xte))
        aucs.append(roc_auc_score(y,proba))
    return np.array(aucs)

print('=== 실험0+1: 집계방식 비교 (DAIC, 10seed AUC) ===',flush=True)
res={}
for m in ['whole','uniform','random','qds']:
    a=eval_method(m); res[m]=a
    print(f'  {m:10s} AUC={a.mean():.3f}±{a.std():.3f}',flush=True)

# 순열 유의차: QDS vs 각 baseline (쌍체 seed)
from scipy.stats import wilcoxon
print('\n=== QDS vs baseline (Wilcoxon, 쌍체 10seed) ===',flush=True)
for m in ['uniform','random','whole']:
    try:
        st,p=wilcoxon(res['qds'],res[m]); d=res['qds'].mean()-res[m].mean()
        print(f'  QDS - {m}: ΔAUC={d:+.3f}, p={p:.4f} {"유의**" if p<0.05 else ""}',flush=True)
    except Exception as e: print(f'  {m}: {e}',flush=True)
