"""
exp49 — Event-anchored Critical Slowing Down (EA-CSD).  ★novelty swing★
가설(북극성): 표정 event 직전 [t*-4s, t*-0.5s]에 lag-1 autocorrelation(AC1)이
  상승(=critical slowing down, 회복력 저하)한다 = within-session precursor.
판정: (a) pre-event AC1 상승 Kendall τ > surrogate (Wilcoxon 단측 p<0.05),
      (b) CMDC·DAIC 두 코퍼스 부호 동일(둘 다 τ>0),  (c) 우울군 CSD 강도 차이.
학습 0. scipy만. 결과: results/exp49_eacsd.csv
"""
import numpy as np, warnings, csv, openpyxl
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import kendalltau, wilcoxon, mannwhitneyu
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data')
FPS=30; PRE0=120; PRE1=15; NSUB=5; MINEV=5   # 4s~0.5s pre-window, 5 sub-windows
AUc=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
AUd=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']

def ac1(x):
    x=x-x.mean(); s=x.std()
    if s<1e-8 or len(x)<4: return 0.0
    a,b=x[1:],x[:-1]
    d=a.std()*b.std()
    return float(((a-a.mean())*(b-b.mean())).mean()/d) if d>1e-12 else 0.0

def detrend(x):
    t=np.arange(len(x)); A=np.polyfit(t,x,1); return x-(A[0]*t+A[1])

def win_tau(s, e):
    """event index e의 pre-window에서 sub-window별 AC1 → time과의 Kendall τ, 마지막 AC1"""
    seg=s[e-PRE0:e-PRE1]
    if len(seg)<NSUB*4: return None
    seg=detrend(seg); L=len(seg)//NSUB; acs=[]
    for k in range(NSUB):
        acs.append(ac1(seg[k*L:(k+1)*L]))
    tau,_=kendalltau(np.arange(NSUB), acs)
    return (0.0 if np.isnan(tau) else tau, acs[-1])

def expressivity(AU):
    Z=(AU-AU.mean(0))/(AU.std(0)+1e-6)
    pc=PCA(1).fit(Z); s=pc.transform(Z)[:,0]
    if np.corrcoef(s, Z.mean(1))[0,1]<0: s=-s   # 방향: 표정활성↑ = s↑
    return s

def subject_csd(AU):
    if len(AU)<PRE0+30: return None
    s=expressivity(AU)
    mad=np.median(np.abs(s-np.median(s)))+1e-6
    ev,_=find_peaks(s, prominence=2.0*mad, distance=45)
    ev=[e for e in ev if e-PRE0>=0 and e+5<len(s)]
    if len(ev)<MINEV: return None
    # 실제 pre-event τ
    taus=[]; lastac=[]
    for e in ev:
        r=win_tau(s,e)
        if r: taus.append(r[0]); lastac.append(r[1])
    if len(taus)<MINEV: return None
    # surrogate: event 근처 아닌 랜덤 위치 (같은 개수)
    evset=np.array(ev); banned=np.zeros(len(s),bool)
    for e in ev: banned[max(0,e-PRE0):min(len(s),e+PRE0)]=True
    cand=[i for i in range(PRE0, len(s)-5) if not banned[i]]
    stau=[]
    if cand:
        step=max(1,len(cand)//max(len(ev),1)); pos=cand[::step][:max(len(ev),1)]
        for e in pos:
            r=win_tau(s,e)
            if r: stau.append(r[0])
    return dict(mean_tau=np.mean(taus), frac_pos=np.mean(np.array(taus)>0),
               last_ac=np.mean(lastac), surr_tau=(np.mean(stau) if stau else 0.0),
               n_ev=len(ev))

def run_corpus(subjAU, y, name):
    R=[]; yy=[]
    for i,au in enumerate(subjAU):
        d=subject_csd(au)
        if d: R.append(d); yy.append(y[i])
    yy=np.array(yy)
    mt=np.array([r['mean_tau'] for r in R]); st=np.array([r['surr_tau'] for r in R])
    print(f'\n===== {name} : CSD 분석대상 {len(R)}명 (우울{yy.sum()}) =====',flush=True)
    # (a) precursor: 실제 τ vs surrogate τ (Wilcoxon 단측 real>surr)
    try: w,p=wilcoxon(mt, st, alternative='greater')
    except: p=1.0
    print(f'  (a) pre-event AC1 상승 τ: real={mt.mean():+.4f} vs surrogate={st.mean():+.4f}',flush=True)
    print(f'      Wilcoxon(real>surr) p={p:.4g}  {"★유의(CSD 존재)" if p<0.05 else "n.s."}',flush=True)
    print(f'      real τ>0 subject 비율={np.mean(mt>0):.2f}',flush=True)
    # (c) 우울 modulation
    md,hc=mt[yy==1],mt[yy==0]
    try: _,pc=mannwhitneyu(md,hc)
    except: pc=1.0
    print(f'  (c) mean_tau  MDD={md.mean():+.4f} HC={hc.mean():+.4f} ({"↑" if md.mean()>hc.mean() else "↓"}우울) MWU p={pc:.3g}',flush=True)
    la=np.array([r['last_ac'] for r in R])
    print(f'      last_AC1  MDD={la[yy==1].mean():.3f} HC={la[yy==0].mean():.3f}',flush=True)
    print(f'      평균 event수={np.mean([r["n_ev"] for r in R]):.1f}',flush=True)
    return dict(name=name,n=len(R),real_tau=float(mt.mean()),surr_tau=float(st.mean()),
                p_csd=float(p),frac_pos=float(np.mean(mt>0)),
                tau_mdd=float(md.mean()),tau_hc=float(hc.mean()),p_mod=float(pc))

# ---- CMDC ----
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
    A=[];Y=[]
    for s,l in cl.items():
        parts=[cq(s,q) for q in range(1,13)]; parts=[p for p in parts if p is not None]
        if parts: A.append(np.vstack(parts));Y.append(l)
    return A,Y
# ---- DAIC ----
def load_daic():
    D=B/'DAIC_WOZ';dl={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
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
    A=[];Y=[]
    for pid,l in dl.items():
        au=dau(pid)
        if au is not None: A.append(au);Y.append(l)
    return A,Y

res=[]
cA,cY=load_cmdc(); res.append(run_corpus(cA,cY,'CMDC'))
dA,dY=load_daic(); res.append(run_corpus(dA,dY,'DAIC'))
# 판정
print('\n========== 판정 ==========',flush=True)
a_ok=all(r['p_csd']<0.05 for r in res)
b_ok=all(r['real_tau']>0 for r in res)
print(f'  (a) 두 코퍼스 CSD 유의(p<0.05): {a_ok}',flush=True)
print(f'  (b) 두 코퍼스 τ>0 부호일치: {b_ok}  (CMDC {res[0]["real_tau"]:+.3f}, DAIC {res[1]["real_tau"]:+.3f})',flush=True)
print(f'  → EA-CSD precursor {"확립 ★NOVELTY 후보★" if (a_ok and b_ok) else "미확립"}',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp49_eacsd.csv','w') as f:
    f.write('corpus,n,real_tau,surr_tau,p_csd,frac_pos,tau_mdd,tau_hc,p_mod\n')
    for r in res:f.write(f"{r['name']},{r['n']},{r['real_tau']:.4f},{r['surr_tau']:.4f},{r['p_csd']:.4g},{r['frac_pos']:.3f},{r['tau_mdd']:.4f},{r['tau_hc']:.4f},{r['p_mod']:.4g}\n")
print('DONE → exp49_eacsd.csv',flush=True)
