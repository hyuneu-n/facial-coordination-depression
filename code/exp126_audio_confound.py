"""exp126 — 오디오가 제2의 confound인가? 우리 자신에게 같은 잣대 적용.
 A) acoustic 영상별 요약(mean+std)만으로 유저 분류 (로지스틱)
 B) 시각 협응만 (오디오 제거, = ablation no_audio 재확인)
 C) 오디오를 유저 내 표준화(=녹음환경 오프셋 제거)하면 성능이 남는가?
"""
import pickle, numpy as np, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
CACHE=os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/cache/mud3_d20.pkl')
with open(CACHE,'rb') as f: C=pickle.load(f)
X={'train':[], 'test':[]}; Y={'train':[], 'test':[]}
Xz={'train':[], 'test':[]}
for n,v in C.items():
    sp=v['split']
    if sp not in ('train','test'): continue
    a=v['aud'][:360]                       # [K, 50] = 영상별 acoustic mean+std
    X[sp].append(np.concatenate([a.mean(0), a.std(0)]))     # 유저 요약 100dim
    az=(a-a.mean(0))/(a.std(0)+1e-6)                        # 유저 내 표준화
    Xz[sp].append(np.concatenate([az.mean(0), az.std(0)]))
    Y[sp].append(v['label'])
for k in X: X[k]=np.array(X[k]); Xz[k]=np.array(Xz[k]); Y[k]=np.array(Y[k])
print('train',X['train'].shape,'test',X['test'].shape, flush=True)

def run(tag, Xtr, Xte):
    sc=StandardScaler().fit(Xtr)
    m=LogisticRegression(max_iter=4000, class_weight='balanced', C=0.1).fit(sc.transform(Xtr), Y['train'])
    p=m.predict(sc.transform(Xte)); pr=m.predict_proba(sc.transform(Xte))[:,1]
    print(f'{tag:44s} F1={f1_score(Y["test"],p):.4f} acc={accuracy_score(Y["test"],p):.4f} AUC={roc_auc_score(Y["test"],pr):.4f}', flush=True)

print()
run('A) acoustic 요약만 (로지스틱, 100dim)', X['train'], X['test'])
run('C) acoustic 유저내 표준화 후 (오프셋 제거)', Xz['train'], Xz['test'])
print()
print('[비교] 얼굴위치·크기만=0.6875 | baseline raw=0.6717 | baseline norm=0.3341', flush=True)
print('[비교] 우리 full=0.7848 | no_audio=0.6666 | no_coord=0.7529', flush=True)
print('DONE', flush=True)
