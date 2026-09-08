"""
exp55 — 경로C: 교차코퍼스 coupling 전이를 RPA(Riemannian Procrustes Analysis)로 해결.
연구질문: exp41의 단순 recenter는 실패. recenter→scale→rotate(RPA) 3단계 정렬이
  zero-shot 전이를 살리는가? 어느 단계가 결정적인가?
프로토콜(누수차단): source 전체 라벨 + target-train 라벨로 정렬/분류 fit → target-test 예측.
  ablation: src_only(zero-shot) / +center / +center+scale / +RPA(rotate).
  참고: target_only(타깃 내부 CV 상한).
공통 14 AU. 결과: results/exp55_rpa.csv
"""
import numpy as np, warnings, csv, glob, openpyxl
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.covariance import ledoit_wolf
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import encode_domains, TLCenter, TLScale, TLRotate
warnings.filterwarnings('ignore')
B=Path('/home/hyuneun/disk_b/🟡facial-prodrome/data'); SEEDS=5
COM=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU25_r','AU26_r']  # 공통 14
def cov_mat(AU):
    seg=(AU-AU.mean(0))/(AU.std(0)+1e-6); c,_=ledoit_wolf(seg); return c+1e-4*np.eye(AU.shape[1])
def base_clf():
    return make_pipeline(TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
def tl_steps(method):
    s=[]
    if method in('center','scale','rpa'): s.append(TLCenter(target_domain='tgt'))
    if method in('scale','rpa'): s.append(TLScale(target_domain='tgt',centered_data=True))
    if method=='rpa': s.append(TLRotate(target_domain='tgt'))
    return s
def transfer(Xs,ys,Xt,yt,method):
    Xs=np.array(Xs);Xt=np.array(Xt);ys=np.array(ys);yt=np.array(yt); aucs=[]
    for seed in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=seed); pb=np.zeros(len(yt))
        for tr,te in skf.split(Xt,yt):
            if method=='src_only':
                clf=base_clf().fit(Xs,ys); pb[te]=clf.predict_proba(Xt[te])[:,1]; continue
            Xf=np.concatenate([Xs,Xt[tr]]); yf=np.concatenate([ys,yt[tr]]); df=np.array(['src']*len(ys)+['tgt']*len(tr))
            Xe,ye=encode_domains(Xf,yf,df)
            steps=tl_steps(method)
            if steps:
                tl=make_pipeline(*steps); tl.fit(Xe,ye); Xf_al=tl.transform(Xe)
                Xte_e,_=encode_domains(Xt[te],yt[te],np.array(['tgt']*len(te))); Xte_al=tl.transform(Xte_e)
            else:
                Xf_al=Xf; Xte_al=Xt[te]
            clf=base_clf().fit(Xf_al,yf); pb[te]=clf.predict_proba(Xte_al)[:,1]
        aucs.append(roc_auc_score(yt,pb))
    return np.mean(aucs),np.std(aucs)
def target_only(Xt,yt):
    Xt=np.array(Xt);yt=np.array(yt);a=[]
    for seed in range(SEEDS):
        skf=StratifiedKFold(5,shuffle=True,random_state=seed);pb=np.zeros(len(yt))
        for tr,te in skf.split(Xt,yt):
            clf=base_clf().fit(Xt[tr],yt[tr]);pb[te]=clf.predict_proba(Xt[te])[:,1]
        a.append(roc_auc_score(yt,pb))
    return np.mean(a)
def run_pair(src,tgt,data,out):
    Xs,ys=data[src];Xt,yt=data[tgt]
    print(f'\n===== {src} → {tgt}  (src n={len(ys)}, tgt n={len(yt)}) =====',flush=True)
    to=target_only(Xt,yt); print(f'  [참고] target_only 상한 = {to:.3f}',flush=True)
    res={}
    for m in ['src_only','center','scale','rpa']:
        r=transfer(Xs,ys,Xt,yt,m); res[m]=r[0]
        tag='(zero-shot)' if m=='src_only' else ''
        print(f'  {m:9s} {r[0]:.3f}±{r[1]:.3f} {tag}',flush=True)
    mark='★RPA가 zero-shot 살림' if res['rpa']>res['src_only']+0.05 else ''
    print(f'  → RPA Δ(vs zero-shot)={res["rpa"]-res["src_only"]:+.3f} {mark}',flush=True)
    out.append([src,tgt,len(ys),len(yt),to,res['src_only'],res['center'],res['scale'],res['rpa']])
# ---- 로더 (공통 14 AU) ----
def load_cmdc():
    C=B/'CMDC/extracted'; wb=openpyxl.load_workbook(C/'SubjectInfo.xlsx');ws=wb.active
    rows=list(ws.iter_rows(values_only=True));hd=list(rows[0]);iID,iMDD=hd.index('ID'),hd.index('MDD')
    cl={str(r[iID]).strip():int(r[iMDD]) for r in rows[1:] if r[iID] is not None}
    def cq(s,q):
        f=C/s/f'Q{q}.csv'
        if not f.exists():return None
        h=[x.strip() for x in open(f).readline().split(',')]
        try:oi=h.index('success');ai=[h.index(c) for c in COM]
        except:return None
        fe=[[float(v[i]) for i in ai] for v in (ln.split(',') for ln in open(f).readlines()[1:]) if len(v)>max(ai) and v[oi] and int(float(v[oi]))==1]
        return np.array(fe) if fe else None
    X=[];Y=[]
    for s,l in cl.items():
        parts=[cq(s,q) for q in range(1,13)];parts=[p for p in parts if p is not None]
        if parts and sum(len(p) for p in parts)>=60: X.append(cov_mat(np.vstack(parts)));Y.append(l)
    return X,Y
def load_daic():
    D=B/'DAIC_WOZ';dl={}
    for f in ['train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv']:
        p=D/f
        if p.exists():
            for r in csv.DictReader(open(p)):dl[r['Participant_ID'].strip()]=int(float(r['PHQ8_Binary']))
    def dau(pid):
        p=D/f'{pid}_CLNF_AUs.txt'
        if not p.exists():return None
        h=[x.strip() for x in open(p).readline().split(',')];oi=h.index('success')
        try:ai=[h.index(c) for c in COM]
        except:return None
        fe=[]
        for ln in open(p).readlines()[1:]:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1:continue
                fe.append([float(v[i]) for i in ai])
            except:pass
        return np.array(fe) if fe else None
    X=[];Y=[]
    for pid,l in dl.items():
        au=dau(pid)
        if au is not None and len(au)>=60: X.append(cov_mat(au));Y.append(l)
    return X,Y
def load_lmvd():
    V=B/'LMVD/extracted/Video_feature'
    def lab(i):
        if (1<=i<=601) or (1117<=i<=1423): return 1
        if (602<=i<=1116) or (1425<=i<=1824): return 0
        return None
    X=[];Y=[]
    for f in sorted(glob.glob(str(V/'*.csv'))):
        l=lab(int(Path(f).stem))
        if l is None:continue
        h=[x.strip() for x in open(f).readline().split(',')]
        try:ci=h.index('confidence');oi=h.index('success');ai=[h.index(c) for c in COM]
        except:continue
        fe=[]
        for ln in open(f).readlines()[1:]:
            v=ln.split(',')
            try:
                if int(float(v[oi]))!=1 or float(v[ci])<0.9:continue
                fe.append([float(v[i]) for i in ai])
            except:pass
        if len(fe)>=60: X.append(cov_mat(np.array(fe)));Y.append(l)
    return X,Y
out=[]
print('로딩 CMDC/DAIC...',flush=True)
data={'CMDC':load_cmdc(),'DAIC':load_daic()}
run_pair('CMDC','DAIC',data,out); run_pair('DAIC','CMDC',data,out)
print('\n로딩 LMVD...',flush=True)
data['LMVD']=load_lmvd()
for s,t in [('LMVD','CMDC'),('CMDC','LMVD'),('LMVD','DAIC'),('DAIC','LMVD')]:
    run_pair(s,t,data,out)
print('\n판정: RPA(rotate)가 src_only(zero-shot) 대비 여러 pair서 +0.05↑ 일관 향상 →',flush=True)
print('      "기하 정렬로 코퍼스 불문 전이되는 coupling 마커"=substantive 기여.',flush=True)
with open('/home/hyuneun/disk_b/🟡facial-prodrome/results/exp55_rpa.csv','w') as f:
    f.write('src,tgt,n_src,n_tgt,target_only,src_only,center,scale,rpa\n')
    for r in out:f.write(','.join(str(x) for x in r)+'\n')
print('DONE → exp55_rpa.csv',flush=True)
