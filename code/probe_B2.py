"""
probe_B2.py — 가정 B 재검증 (전조 = 앵커 전후 '변화'에 초점)
1) 짧은 윈도우(3s)
2) 정적 평균이 아니라 앵커 전(pre) → 후(post) AU '변화량(delta)'
3) 부정정서 앵커 vs 긍정(happy) 앵커 반응성 대비
4) 반응성 지표: post 구간 AU 변동성(std), 활동량(|diff| 평균 = 표정 움직임)
모델 없이 Mann-Whitney + Cohen's d.
"""
import numpy as np, csv
from pathlib import Path
from scipy.stats import mannwhitneyu

D = Path('/home/hyuneun/disk_b/🟡facial-prodrome/data/DAIC_WOZ')
PRE, POST = 3.0, 4.0
NEG = ['diagnosed with depression','feel guilty','regret','made you feel really badly',
       "don't sleep",'feel in that moment','feeling lately']
POS = ['last time you felt really happy','really happy','proud','best','enjoy']
AU = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r',
      'AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']

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
            try: rows.append((float(r['start_time']),float(r['stop_time']),r['speaker'].strip(),(r['value'] or '').lower()))
            except: pass
    return rows

def anchors(rows,keys):
    return [sp for st,sp,spk,val in rows if spk=='Ellie' and any(k in val for k in keys)]

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

def reactivity(pid,anch):
    """앵커별: post 구간의 (a)변동성 std, (b)표정 움직임 |diff|평균, (c)pre->post 평균변화 |delta|"""
    ts,au=au_series(pid)
    if ts is None or len(ts)==0 or not anch: return None
    R=[]
    for a in anch:
        pre=au[(ts>=a-PRE)&(ts<a)]; post=au[(ts>=a)&(ts<a+POST)]
        if len(post)<4 or len(pre)<2: continue
        stdv=post.std(0).mean()
        motion=np.abs(np.diff(post,axis=0)).mean()
        delta=np.abs(post.mean(0)-pre.mean(0)).mean()
        R.append([stdv,motion,delta])
    if not R: return None
    return np.array(R).mean(0)  # [std, motion, delta]

def d(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<2 or len(b)<2: return np.nan
    sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/sp if sp>0 else np.nan

def run(name,keys,lab):
    X,y=[],[]
    for pid,l in lab.items():
        r=reactivity(pid,anchors(transcript(pid),keys))
        if r is not None: X.append(r); y.append(l)
    X=np.array(X); y=np.array(y); dep,nor=X[y==1],X[y==0]
    print(f'\n=== [{name}] n={len(y)} (우울{int(y.sum())}/정상{int((y==0).sum())}) ===',flush=True)
    for i,m in enumerate(['post_변동성','표정움직임','pre->post_변화']):
        dd=d(dep[:,i],nor[:,i])
        try: _,p=mannwhitneyu(dep[:,i],nor[:,i],alternative='two-sided')
        except: p=np.nan
        s='***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        print(f'  {m:16s} d={dd:+.2f} p={p:.3f}{s}  우울={dep[:,i].mean():.3f} 정상={nor[:,i].mean():.3f}',flush=True)

lab=labels()
run('부정정서 앵커',NEG,lab)
run('긍정(happy) 앵커',POS,lab)
