"""exp122 — MUD3 leakage 점검. 얼굴/신호 내용 없이 '메타 정보'만으로 얼마나 맞히나?
 A) 영상 개수만  B) raw 픽셀 얼굴 스케일·위치만  C) 프레임 길이 통계만
비교기준: baseline F1 0.689, 우리 0.747
"""
import pickle, numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
R = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
lab = pd.read_csv(f'{R}/labels.csv'); split_of = dict(zip(lab['names'], lab['split']))

rows = []
for fn, y in [('dep_feat.pkl',1), ('nondep_feat.pkl',0)]:
    with open(f'{R}/{fn}','rb') as f: d = pickle.load(f)
    for name, vids in zip(d['name'], d['features']):
        sp = split_of.get(name)
        if sp is None: continue
        n = len(vids)
        fl = np.array([np.asarray(v).shape[0] for v in vids], dtype=float)
        # raw 픽셀 얼굴 스케일/위치 (정규화 이전)
        sub = vids[:min(n,50)]
        cx, cy, sc = [], [], []
        for v in sub:
            P = np.asarray(v, dtype=np.float64)[:, :136].reshape(-1,68,2)
            c = P.mean(1); cx.append(c[:,0].mean()); cy.append(c[:,1].mean())
            sc.append(np.sqrt(((P-c[:,None,:])**2).sum(2).mean()))
        rows.append(dict(name=name, y=y, split=sp,
            n_vid=n, log_n=np.log1p(n),
            fl_mean=fl.mean(), fl_med=np.median(fl), fl_max=fl.max(), fl_std=fl.std(),
            cx=np.mean(cx), cy=np.mean(cy), scale=np.mean(sc), scale_std=np.std(sc)))
    del d
df = pd.DataFrame(rows)
print(f'users={len(df)}', flush=True)

FEATS = {
 'A) 영상 개수만':            ['n_vid','log_n'],
 'B) raw 얼굴 스케일·위치만': ['cx','cy','scale','scale_std'],
 'C) 프레임 길이 통계만':     ['fl_mean','fl_med','fl_max','fl_std'],
 'A+B+C 전부(메타 총합)':     ['n_vid','log_n','cx','cy','scale','scale_std','fl_mean','fl_med','fl_max','fl_std'],
}
tr, te = df[df.split=='train'], df[df.split=='test']
print(f"\n{'조합':28s} {'test F1':>9s} {'acc':>7s} {'AUC':>7s}", flush=True)
for tag, cols in FEATS.items():
    sc = StandardScaler().fit(tr[cols])
    m = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(tr[cols]), tr.y)
    p = m.predict(sc.transform(te[cols])); pr = m.predict_proba(sc.transform(te[cols]))[:,1]
    print(f'{tag:28s} {f1_score(te.y,p):9.4f} {accuracy_score(te.y,p):7.4f} {roc_auc_score(te.y,pr):7.4f}', flush=True)

print('\n[집단 통계 train]', flush=True)
g = df[df.split=='train'].groupby('y')[['n_vid','scale','cx','cy','fl_med']].mean()
print(g.to_string(), flush=True)
print('\n[비교] baseline F1=0.689 / 우리 F1=0.747', flush=True)
print('DONE', flush=True)
