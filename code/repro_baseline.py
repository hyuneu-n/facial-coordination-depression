"""MUD3 baseline(LSTM_han) 재현 — 데이터 1회 로드 + multi-seed + 모델선택 기준(val loss vs val F1) 비교.
저자 run_rnn.sh 설정: epochs 20, bs 2, lr 1e-5, sched None, alpha 30.
"""
import sys, os, pickle, random, time, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/CCAC-baseline'))
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from models import LSTM_han

R = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/MUD3/raw')
DEV = 'cuda'; EPOCHS = 20; BS = 2; LR = 1e-5; ALPHA = 30.0

# ---- 데이터 1회 로드 (baseline과 동일 전처리: 60프레임 cut, 360영상 cut) ----
lab = pd.read_csv(f'{R}/labels.csv'); split_of = dict(zip(lab['names'], lab['split']))
data = {'train': [], 'val': [], 'test': []}
for fn, y in [('dep_feat.pkl', 1), ('nondep_feat.pkl', 0)]:
    with open(f'{R}/{fn}', 'rb') as f: d = pickle.load(f)
    for name, vids in zip(d['name'], d['features']):
        sp = split_of.get(name)
        if sp is None: continue
        data[sp].append(([np.asarray(v)[:60, :] for v in vids[:360]], y, name))
    del d
print({k: (len(v), sum(y for _, y, _ in v)) for k, v in data.items()}, flush=True)

def collate(batch):
    feats, lengths = [], []
    for vids, _, _ in batch:
        ts = []
        for f in vids:
            f = torch.from_numpy(np.asarray(f, dtype=np.float32))
            if f.shape[0] < 60: f = torch.cat([f, torch.zeros(60 - f.shape[0], f.shape[1])], 0)
            if f.shape[1] < 161: f = torch.cat([f, torch.zeros(f.shape[0], 161 - f.shape[1])], 1)
            ts.append(f[:60, :161])
        lengths.append(len(ts)); feats.append(pad_sequence(ts, batch_first=True))
    x = pad_sequence(feats, batch_first=True)
    y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return (x, lengths, [b[2] for b in batch]), y

def batches(split, bs, shuffle):
    idx = list(range(len(data[split])))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield collate([data[split][j] for j in idx[i:i+bs]])

def evaluate(net, split, lf):
    net.eval(); P, Y, L, n = [], [], 0.0, 0
    with torch.no_grad():
        for (x, ln, nm), y in batches(split, BS, False):
            logits = net((x.to(DEV), ln, nm)).view(-1, 1)
            yy = y.to(DEV).view(-1, 1)
            L += lf(logits, yy).item() * yy.numel(); n += yy.numel()
            P += (torch.sigmoid(logits) > 0.5).view(-1).long().cpu().tolist()
            Y += yy.view(-1).long().cpu().tolist()
    return dict(loss=L/n, acc=accuracy_score(Y, P), f1=f1_score(Y, P, zero_division=0),
                p=precision_score(Y, P, zero_division=0), r=recall_score(Y, P, zero_division=0))

res = []
for seed in [110, 42, 7, 0, 1, 2, 3, 4, 5, 6]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net = LSTM_han(d=256, t_downsample=4).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss()
    best = {'loss': (1e9, None), 'f1': (-1, None)}
    for ep in range(EPOCHS):
        net.train(); t0 = time.time()
        for (x, ln, nm), y in batches('train', BS, True):
            opt.zero_grad()
            loss = lf(net((x.to(DEV), ln, nm)).view(-1, 1), y.to(DEV).view(-1, 1))
            loss.backward(); opt.step()
        v = evaluate(net, 'val', lf)
        if v['loss'] <= best['loss'][0]: best['loss'] = (v['loss'], {k: t.detach().clone() for k, t in net.state_dict().items()})
        if v['f1'] >= best['f1'][0]:  best['f1'] = (v['f1'], {k: t.detach().clone() for k, t in net.state_dict().items()})
        print(f"seed{seed} ep{ep} val loss={v['loss']:.4f} f1={v['f1']:.4f} acc={v['acc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    for crit in ['loss', 'f1']:
        net.load_state_dict(best[crit][1])
        t = evaluate(net, 'test', lf)
        print(f"  >> seed{seed} select-by-val-{crit}: TEST f1={t['f1']:.4f} acc={t['acc']:.4f} p={t['p']:.4f} r={t['r']:.4f}", flush=True)
        res.append((seed, crit, t['f1'], t['acc']))

print('\n===== MUD3 baseline 재현 요약 (논문 F1 ~0.77) =====', flush=True)
for crit in ['loss', 'f1']:
    f = np.array([r[2] for r in res if r[1] == crit]); a = np.array([r[3] for r in res if r[1] == crit])
    print(f'  select-by-val-{crit}: test F1 = {f.mean():.4f} ± {f.std():.4f}   acc = {a.mean():.4f} ± {a.std():.4f}  (n={len(f)})', flush=True)
print('DONE', flush=True)
