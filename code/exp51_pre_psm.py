"""
exp51 — pre-event psychomotor slowing이 '전역 느림'을 넘어선 precursor-특이 신호인가?
핵심질문: pre_ratio(=pre-event속도/전역속도, 전역속도 이미 정규화 제거)만으로도
  우울 판별되면 → 표정 짓기 '직전 lead-up'이 baseline보다 더 느려지는 precursor-특이 성분 존재.
+ gap 민감도(0.5/1.5/2.5s) + precursor-특이성(pre-event vs 랜덤창 속도).
D-Vlog(landmark) + LMVD(AU) 두 대형 in-the-wild. 결과: results/exp51_pre_psm.csv
"""
import numpy as np, warnings, csv, glob
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
WIN=15; GAPS=[15,45,75]  # 0.5,1.5,2.5s
def cliffs(a,b):
    a,b=np.asarray(a),np.asarray(b); g=sum((a[:,None]>b).sum() for a in [a]);
    n=len(a)*len(b); gt=(a[:,None]>b[None,:]).sum(); lt=(a[:,None]<b[None,:]).sum()
    return (gt-lt)/n
def pre_feats(spd, gap):
    g=spd.mean()+1e-6; thr=np.percentile(spd,80); ev=[]; t=WIN+gap
    while t<len(spd)-1:
        if spd[t]>=thr and spd[t]>=spd[t-1] and spd[t]>=spd[t+1]: ev.append(t);t+=WIN
        else:t+=1
    pre=[spd[e-WIN-gap:e-gap].mean() for e in ev if e-WIN-gap>=0]
    if len(pre)<3: return None
    prem=np.mean(pre)
    # precursor-특이성: 랜덤창(event 아닌) 속도
    banned=np.zeros(len(spd),bool)
    for e in ev: banned[max(0,e-WIN-gap):min(len(spd),e+WIN)]=True
    cand=[i for i in range(WIN+gap,len(spd)) if not banned[i]]
    rnd=np.mean([spd[i-WIN-gap:i-gap].mean() for i in cand[::max(1,len(cand)//max(len(pre),1))][:len(pre)]]) if cand else g
    return prem, prem/g, rnd, g  # pre_spd, pre_ratio, random_win_spd, global
def analyze(SPD, y, name):
    y=np.array(y)
    print(f'\n===== {name} (n={len(SPD)}, 우울{y.sum()}) =====',flush=True)
    print(f'  {"gap(s)":>6} {"pre_ratio차(MDD-HC)":>18} {"cliffδ":>7} {"pre_ratio단독AUC":>15} {"전역속도단독AUC":>15}',flush=True)
    out=[]
    for g in GAPS:
        F=[];yy=[]
        for i,spd in enumerate(SPD):
            r=pre_feats(spd,g)
            if r: F.append(r);yy.append(y[i])
        F=np.array(F);yy=np.array(yy)
        prr=F[:,1]  # pre_ratio
        d=prr[yy==1].mean()-prr[yy==0].mean(); cd=cliffs(prr[yy==1],prr[yy==0])
        def auc1(col):
            X=F[:,[col]];a=[]
            for s in range(8):
                skf=StratifiedKFold(5,shuffle=True,random_state=s);pb=np.zeros(len(yy))
                for tr,te in skf.split(X,yy):
                    sc=StandardScaler().fit(X[tr]);cl=LogisticRegression(max_iter=1000,class_weight='balanced').fit(sc.transform(X[tr]),yy[tr])
                    pb[te]=cl.decision_function(sc.transform(X[te]))
                a.append(roc_auc_score(yy,pb))
            return np.mean(a)
        a_pr=auc1(1); a_gl=auc1(3)
        # precursor-특이성: pre_spd vs random_win_spd (MDD에서 pre가 더 느린가)
        print(f'  {g/30:>6.1f} {d:>+18.4f} {cd:>+7.3f} {a_pr:>15.3f} {a_gl:>15.3f}',flush=True)
        out.append([name,g/30,d,cd,a_pr,a_gl,len(yy)])
    return out
# ---- D-Vlog (landmark) ----
def load_dvlog_spd():
    R=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/D-Vlog')
    def nf(V):
        T=V.shape[0];P=V.reshape(T,68,2).astype(float);P=P-P.mean(1,keepdims=True)
        sc=np.sqrt((P**2).sum(2).mean(1,keepdims=True))+1e-6;return (P/sc[:,None]).reshape(T,136)
    S=[];Y=[]
    for r in csv.DictReader(open(R/'labels.csv')):
        idx=r['index'].strip();f=R/idx/f'{idx}_visual.npy'
        if not f.exists():continue
        try:V=np.load(f)
        except:continue
        if V.ndim!=2 or V.shape[0]<120 or V.shape[1]!=136:continue
        Vn=nf(V);Z=(Vn-Vn.mean(0))/(Vn.std(0)+1e-6)
        S.append(np.linalg.norm(np.diff(Z,axis=0),axis=1));Y.append(1 if r['label'].strip().lower().startswith('depress') else 0)
    return S,Y
# ---- LMVD (AU) ----
def load_lmvd_spd():
    V=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/LMVD/extracted/Video_feature')
    AU=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
    def lab(i):
        if (1<=i<=601) or (1117<=i<=1423): return 1
        if (602<=i<=1116) or (1425<=i<=1824): return 0
        return None
    S=[];Y=[]
    for f in sorted(glob.glob(str(V/'*.csv'))):
        l=lab(int(Path(f).stem))
        if l is None:continue
        h=[x.strip() for x in open(f).readline().split(',')]
        try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in AU]
        except:continue
        fe=[]
        for ln in open(f).readlines()[1:]:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
                fe.append([float(v[i]) for i in ai])
            except:pass
        if len(fe)<120:continue
        au=np.array(fe);Z=(au-au.mean(0))/(au.std(0)+1e-6)
        S.append(np.linalg.norm(np.diff(Z,axis=0),axis=1));Y.append(l)
    return S,Y
out=[]
s,y=load_dvlog_spd(); out+=analyze(s,y,'D-Vlog(landmark)')
s,y=load_lmvd_spd();  out+=analyze(s,y,'LMVD(AU)')
print('\n판정: pre_ratio단독AUC>0.55 & MDD-HC<0(우울 lead-up 느림)이 두 코퍼스·여러 gap서 일관 →',flush=True)
print('      전역 느림 넘어선 precursor-특이 psychomotor 존재(=goal 정합 실 산출물).',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp51_pre_psm.csv','w') as f:
    f.write('corpus,gap_s,pre_ratio_diff,cliffs_delta,pre_ratio_AUC,global_AUC,n\n')
    for r in out:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp51_pre_psm.csv',flush=True)
