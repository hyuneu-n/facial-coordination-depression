"""exp137 — MGMN(SIGIR'26) 핵심 메커니즘(raw feature 코사인유사도 그래프) confound 취약성 검증.
MGMN Stage1: S_V = l2norm(X_V) l2norm(X_V)^T (raw visual/acoustic 특징의 코사인유사도로 그래프 구성).
그래프 유사도 기반 분류(kNN-cosine, LabelPropagation)를 raw vs confound-free 조건에서 비교.
목적: "우리 confound-free 경량모델의 강건성"을 뒷받침하는 대조 실험 (비판이 아니라 우리 방법 근거).
"""
import os, pickle, numpy as np, pandas as pd, warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')

R = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
ALIGNED = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_aligned.pkl')
lab = pd.read_csv(f'{R}/labels.csv'); split_of = dict(zip(lab['names'], lab['split']))

# ---- (A) RAW: MGMN이 실제로 받는 것과 동일한 형태 — 정규화 없는 유저 평균 landmark+acoustic ----
raw_vis, raw_aud, Y, SP, names = [], [], [], [], []
for fn, y in [('dep_feat.pkl',1), ('nondep_feat.pkl',0)]:
    with open(f'{R}/{fn}','rb') as f: d = pickle.load(f)
    for n, vids in zip(d['name'], d['features']):
        sp = split_of.get(n)
        if sp is None: continue
        allf = np.concatenate([np.asarray(v, dtype=np.float64) for v in vids], axis=0)  # [전체프레임, 161]
        raw_vis.append(allf[:, :136].mean(0)); raw_aud.append(allf[:, 136:].mean(0))
        Y.append(y); SP.append(sp); names.append(n)
    del d
RV = np.array(raw_vis); RA = np.array(raw_aud); Y = np.array(Y); SP = np.array(SP)
print(f'RAW: users={len(Y)}', flush=True)

# ---- (B) confound-free: Procrustes정렬 landmark(PCA20) + 유저내표준화 오디오 ----
C = pickle.load(open(ALIGNED,'rb'))
CV, CA = [], []
for n in names:
    v = C[n]
    m = v['mean'].mean(0)                                    # pose-free 표정 평균
    a = v['aud']; az = (a - a.mean(0)) / (a.std(0) + 1e-6)    # 유저내 표준화(녹음환경 제거)
    CV.append(m); CA.append(az.mean(0))
CV = np.array(CV); CA = np.array(CA)
print(f'CONFOUND-FREE: users={len(CV)}', flush=True)

def cosine_graph_eval(Xv, Xa, tag):
    """MGMN Stage1 재현: l2norm 후 코사인유사도 → graph 기반 분류(kNN-cosine, LabelPropagation)"""
    Xv = normalize(Xv); Xa = normalize(Xa)                     # MGMN Eq: unit hypersphere 투영
    X = np.concatenate([Xv, Xa], axis=1)                       # S_mut = alpha*S_V+(1-alpha)*S_A 근사(단순결합)
    tr, te = SP != 'test', SP == 'test'
    out = {}
    for k in [5, 15, 30]:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance').fit(X[tr], Y[tr])
        p = knn.predict(X[te]); pr = knn.predict_proba(X[te])[:,1]
        out[f'kNN(k={k})'] = (f1_score(Y[te],p,zero_division=0), accuracy_score(Y[te],p), roc_auc_score(Y[te],pr))
    lp = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
    ylp = Y.copy(); ylp[te] = -1
    lp.fit(X, ylp)
    p = lp.transduction_[te]
    out['LabelProp'] = (f1_score(Y[te],p,zero_division=0), accuracy_score(Y[te],p), float('nan'))
    print(f'\n[{tag}]', flush=True)
    for k,(f1,acc,auc) in out.items():
        print(f'  {k:14s} F1={f1:.4f} acc={acc:.4f} AUC={auc:.4f}', flush=True)
    return out

r_raw = cosine_graph_eval(RV, RA, 'RAW (MGMN이 받는 원 조건, confound 포함)')
r_cf  = cosine_graph_eval(CV, CA, 'CONFOUND-FREE (위치·크기·회전 제거 + 오디오 유저내표준화)')

print('\n===== 요약: raw-유사도그래프의 confound 의존성 =====', flush=True)
for k in r_raw:
    print(f'  {k:14s}  raw F1={r_raw[k][0]:.4f}  →  confound-free F1={r_cf[k][0]:.4f}  (Δ={r_raw[k][0]-r_cf[k][0]:+.4f})', flush=True)
print('\n[비교] 우리 confound-free 경량모델(exp134, GRU 시퀀스): F1=0.7534(+오디오) / 0.6826(표정만)', flush=True)
print('[비교] MGMN 논문 보고 MUD3: F1=95.1% (raw feature 코사인유사도 그래프 기반)', flush=True)
print('DONE', flush=True)
