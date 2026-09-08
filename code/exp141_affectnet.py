"""exp141 — AffectNet 독립검증: D-Vlog 판별부위(입·턱, AU15-17)가 일반 정서표현에서도
sad vs happy를 가장 잘 가르는지 확인. labels.csv를 정답으로 사용(폴더명 아님, 불일치 존재).
static 이미지라 '협응'(시간) 아닌 '형태(위치)' 판별력만 검증 — 스코프 명시.
"""
import os, glob, numpy as np, pandas as pd, warnings
import dlib, cv2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings('ignore')

R = os.path.expanduser('~/disk_b/🟡facial-prodrome/data/AffectNet')
lab = pd.read_csv(f'{R}/labels.csv')
print(f'labels.csv 총 {len(lab)}행, 라벨분포:\n{lab.label.value_counts()}', flush=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/tmp/sp68.dat')

def get_landmarks(path):
    img = cv2.imread(path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    faces = detector(gray, 1)
    if len(faces) == 0:
        rect = dlib.rectangle(0, 0, w, h)  # 이미 크롭된 얼굴로 가정
    else:
        rect = max(faces, key=lambda r: r.width()*r.height())
    shp = predictor(gray, rect)
    return np.array([[p.x, p.y] for p in shp.parts()], dtype=np.float64)

def nf(P):  # 위치·크기 정규화
    c = P.mean(0); Pc = P - c
    sc = np.sqrt((Pc**2).sum(1).mean()) + 1e-6
    return Pc / sc

TARGET = {'sad': 800, 'happy': 800, 'neutral': 400, 'anger': 400, 'surprise': 400}
rows = []
rng = np.random.RandomState(0)
t0 = __import__('time').time()
for emo, ncap in TARGET.items():
    sub = lab[lab.label == emo]
    idx = rng.permutation(len(sub))[:ncap]
    ok = 0
    for i in idx:
        rel = sub.iloc[i]['pth']
        # pth는 Train 기준 상대경로(폴더/파일) — 실제 파일이 Train 하위 어디 있는지 폴더명 무관하게 탐색
        cand = f"{R}/Train/{rel}"
        if not os.path.exists(cand):
            fname = os.path.basename(rel)
            hits = glob.glob(f"{R}/Train/*/{fname}")
            if not hits: continue
            cand = hits[0]
        P = get_landmarks(cand)
        if P is None: continue
        rows.append(dict(emo=emo, lm=nf(P)))
        ok += 1
    print(f'  {emo}: {ok}/{len(idx)} landmark 성공 ({__import__("time").time()-t0:.0f}s)', flush=True)

print(f'\n총 샘플 {len(rows)}개', flush=True)
Y = np.array([r['emo'] for r in rows])
LM = np.stack([r['lm'] for r in rows])  # [N,68,2]

REGIONS = {
    '입(48-67,AU15-17 포함)': list(range(48,68)),
    '턱선(0-16)': list(range(0,17)),
    '눈썹(17-26)': list(range(17,27)),
    '코(27-35)': list(range(27,36)),
    '눈(36-47)': list(range(36,48)),
    '전체(0-67)': list(range(0,68)),
}

def eval_pair(a, b, region_idx):
    m = np.isin(Y, [a,b])
    X = LM[m][:, region_idx].reshape(m.sum(), -1)
    y = (Y[m] == a).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(Xtr), ytr)
    pr = clf.predict_proba(sc.transform(Xte))[:,1]
    return roc_auc_score(yte, pr)

print('\n[부위별 sad vs happy 판별력 (AUC, 정규화 landmark)]', flush=True)
res = {}
for name, idx in REGIONS.items():
    auc = eval_pair('sad','happy', idx)
    res[name] = auc
    print(f'  {name:24s} AUC={auc:.4f}', flush=True)
rank = sorted(res.items(), key=lambda x:-x[1])
print(f'\n입 영역 순위: {[n for n,_ in rank].index("입(48-67,AU15-17 포함)")+1} / {len(rank)}', flush=True)

print('\n[대조: sad vs neutral, sad vs anger — 입 영역이 일관되게 상위인가]', flush=True)
for a,b in [('sad','neutral'), ('sad','anger')]:
    print(f' --- {a} vs {b} ---', flush=True)
    r2 = {name: eval_pair(a,b,idx) for name,idx in REGIONS.items()}
    for name,auc in sorted(r2.items(), key=lambda x:-x[1]):
        print(f'    {name:24s} AUC={auc:.4f}', flush=True)
print('DONE', flush=True)
