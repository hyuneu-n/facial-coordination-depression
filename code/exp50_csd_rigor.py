"""
exp50 — EA-CSD 아티팩트 검증(앞단 의심). exp49의 pre-event AC1 상승이
진짜 anticipatory slowing인가, 표정 onset ramp가 새어든 deterministic 아티팩트인가?
검증1(결정적): gap 민감도 — pre-window(3s폭)를 event에서 δ=0.5→3.5s 밀어냄.
   큰 gap서도 τ>0 유지 = 진짜 / gap↑에 소멸 = onset ramp 아티팩트.
검증2: post-event 대칭성 — 대칭 bump면 τ_post≈-τ_pre(peak서 AC1 최대).
결과: results/exp50_csd_rigor.csv
"""
import numpy as np, warnings, csv, openpyxl
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import kendalltau, wilcoxon
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
WIN=90; NSUB=5; MINEV=5; GAPS=[15,45,75,105]  # 0.5,1.5,2.5,3.5s @30fps
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']
def ac1(x):
    x=x-x.mean(); s=x.std()
    if s<1e-8 or len(x)<4: return 0.0
    a,b=x[1:],x[:-1]; d=a.std()*b.std()
    return float(((a-a.mean())*(b-b.mean())).mean()/d) if d>1e-12 else 0.0
def detrend(x):
    t=np.arange(len(x)); A=np.polyfit(t,x,1); return x-(A[0]*t+A[1])
def tau_of(seg):
    if len(seg)<NSUB*4: return None
    seg=detrend(seg); L=len(seg)//NSUB; acs=[ac1(seg[k*L:(k+1)*L]) for k in range(NSUB)]
    t,_=kendalltau(np.arange(NSUB),acs); return 0.0 if np.isnan(t) else t
def expressivity(AU):
    Z=(AU-AU.mean(0))/(AU.std(0)+1e-6); s=PCA(1).fit_transform(Z)[:,0]
    if np.corrcoef(s,Z.mean(1))[0,1]<0: s=-s
    return s
def events(s):
    mad=np.median(np.abs(s-np.median(s)))+1e-6
    ev,_=find_peaks(s,prominence=2.0*mad,distance=45); return ev
def subj_taus(AU,gap):
    """gap에서: real pre-window τ, surrogate(event-free) τ, post-event τ"""
    if len(AU)<WIN+gap+30: return None
    s=expressivity(AU); ev=events(s)
    valid=[e for e in ev if e-gap-WIN>=0 and e+gap+WIN<len(s)]
    if len(valid)<MINEV: return None
    pre=[tau_of(s[e-gap-WIN:e-gap]) for e in valid]; pre=[p for p in pre if p is not None]
    post=[tau_of(s[e+gap:e+gap+WIN][::-1]) for e in valid]; post=[p for p in post if p is not None]  # 뒤집어 event서 멀어지는 방향
    banned=np.zeros(len(s),bool)
    for e in ev: banned[max(0,e-gap-WIN):min(len(s),e+gap+WIN)]=True
    cand=[i for i in range(gap+WIN,len(s)) if not banned[i]]
    sur=[]
    if cand:
        step=max(1,len(cand)//max(len(valid),1))
        for e in cand[::step][:len(valid)]:
            t=tau_of(s[e-gap-WIN:e-gap])
            if t is not None: sur.append(t)
    if len(pre)<MINEV: return None
    return np.mean(pre), (np.mean(sur) if sur else 0.0), (np.mean(post) if post else 0.0)
def load_cmdc():
    C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
    rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
    cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
    def cq(s,q):
        f=C/s/f'Q{q}.csv'
        if not f.exists():return None
        h=[x.strip() for x in open(f).readline().split(',')]
        try:oi=h.index('success');ai=[h.index(c) for c in AUc]
        except:return None
        fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
        return np.array(fe) if fe else None
    A=[]
    for s in cl:
        parts=[cq(s,q) for q in range(1,13)]; parts=[p for p in parts if p is not None]
        if parts: A.append(np.vstack(parts))
    return A
def load_daic():
    D=B/'DAIC_WOZ';ids=[]
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)):ids.append(r['Participant_ID'].strip())
    def dau(pid):
        p=D/f'{pid}_CLNF_AUs.txt'
        if not p.exists():return None
        h=[x.strip() for x in open(p).readline().split(',')];oi=h.index('success');ai=[h.index(c) for c in AUd]
        fe=[]
        for ln in open(p).readlines()[1:]:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1:continue
                fe.append([float(v[i]) for i in ai])
            except:pass
        return np.array(fe) if fe else None
    A=[]
    for pid in ids:
        au=dau(pid)
        if au is not None: A.append(au)
    return A
def run(A,name,out):
    print(f'\n===== {name} (n={len(A)}) — gap 민감도 =====',flush=True)
    print(f'  {"gap(s)":>7} {"real_τ":>8} {"surr_τ":>8} {"post_τ":>8} {"p(real>surr)":>13} {"n":>4}',flush=True)
    for g in GAPS:
        rows=[subj_taus(au,g) for au in A]; rows=[r for r in rows if r]
        if len(rows)<10: print(f'  {g/30:>7.1f} (n부족 {len(rows)})',flush=True); continue
        pre=np.array([r[0] for r in rows]);sur=np.array([r[1] for r in rows]);post=np.array([r[2] for r in rows])
        try: _,p=wilcoxon(pre,sur,alternative='greater')
        except: p=1.0
        mark='★' if p<0.05 else ' '
        print(f'  {g/30:>7.1f} {pre.mean():>+8.4f} {sur.mean():>+8.4f} {post.mean():>+8.4f} {p:>13.3g}{mark} {len(rows):>4}',flush=True)
        out.append([name,g/30,pre.mean(),sur.mean(),post.mean(),p,len(rows)])
out=[]
run(load_cmdc(),'CMDC',out)
run(load_daic(),'DAIC',out)
print('\n판정: real_τ가 gap 커져도(3.5s) surrogate 대비 유의 유지 → 진짜 anticipatory slowing.',flush=True)
print('      gap↑에 real_τ→0/유의소멸 → 표정 onset ramp 아티팩트(=EA-CSD 기각).',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp50_csd_rigor.csv','w') as f:
    f.write('corpus,gap_s,real_tau,surr_tau,post_tau,p,n\n')
    for r in out:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp50_csd_rigor.csv',flush=True)
