"""exp116 — 크로스모달 중반성 재현 (LMVD·E-DAIC). 얼굴 활동량-음성 활동량 구간별 연동 MDD vs HC.
D-Vlog(exp115)서 중반(40-80%) 약화. 다른 코퍼스서도? 음성품질 한계는 정직히.
"""
import numpy as np, warnings, os, re, glob, csv
import pandas as pd
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
AUS=['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
DATA='/home/hyuneun/disk_b/🟡facial-prodrome/data/';NSEG=5;NR=250
def rs(x,n=NR):
    x=np.asarray(x,float)
    if len(x)<2:return None
    idx=np.linspace(0,len(x)-1,n);return np.interp(idx,np.arange(len(x)),x)
def seg_coupling(face_act,voice_act):
    f=rs(face_act);v=rs(voice_act)
    if f is None or v is None:return None
    segs=[]
    for s in range(NSEG):
        i0=s*NR//NSEG;i1=(s+1)*NR//NSEG
        a=f[i0:i1];b=v[i0:i1]
        if a.std()<1e-9 or b.std()<1e-9:segs.append(np.nan);continue
        segs.append(abs(np.corrcoef(a,b)[0,1]))
    return segs
def report(name,Seg,Y):
    Seg=np.array(Seg);Y=np.array(Y);ok=~np.isnan(Seg).any(1);Seg=Seg[ok];Y=Y[ok]
    print(f'\n#### {name} n={len(Y)} (MDD={int(Y.sum())}) ####',flush=True)
    for s in range(NSEG):
        a=Seg[Y==1,s];b=Seg[Y==0,s]
        try:_,p=mannwhitneyu(a,b,alternative='two-sided')
        except:p=1
        dr='MDD<HC(약화)' if a.mean()<b.mean() else 'MDD>HC'
        print(f'  {s*20}-{(s+1)*20}%: MDD={a.mean():.3f} HC={b.mean():.3f} [{dr}] p={p:.2e}',flush=True)
# ===== LMVD =====
def lm(i):return 1 if (1<=i<=601 or 1117<=i<=1423) else 0
aud={}
for f in glob.glob(DATA+'LMVD/extracted/Audio_feature/*.npy'):
    m=re.search(r'(\d+)',os.path.basename(f))
    if m:aud[int(m.group(1))]=f
Seg=[];Y=[]
for vf in sorted(glob.glob(DATA+'LMVD/extracted/Video_feature/*.csv')):
    m=re.search(r'(\d+)',os.path.basename(vf))
    if not m:continue
    i=int(m.group(1))
    if i not in aud:continue
    try:df=pd.read_csv(vf,usecols=lambda c:c.strip() in set(['success']+AUS))
    except:continue
    df.columns=[c.strip() for c in df.columns]
    if not set(AUS).issubset(df.columns):continue
    if 'success' in df:df=df[df['success']==1]
    AU=df[AUS].values.astype(float)
    if len(AU)<30:continue
    face=np.concatenate([[0],np.linalg.norm(np.diff(AU,axis=0),axis=1)])  # 얼굴 활동량
    try:V=np.load(aud[i])
    except:continue
    voice=np.linalg.norm(V,axis=1)  # 음성 활동량(VGGish energy)
    sc=seg_coupling(face,voice)
    if sc is not None and not np.isnan(sc).any():Seg.append(sc);Y.append(lm(i))
report('LMVD',Seg,Y)
# ===== E-DAIC =====
lab={}
for spn in ['train','dev','test']:
    p=DATA+f'E-DAIC/labels/{spn}_split.csv'
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            pid=r.get('Participant_ID','').strip();b=r.get('PHQ_Binary','').strip()
            if pid and b in('0','1'):lab[pid]=int(b)
Seg=[];Y=[]
for dfold in sorted(glob.glob(DATA+'E-DAIC/extracted/*_P/')):
    pid=os.path.basename(dfold.rstrip('/')).replace('_P','')
    if pid not in lab:continue
    vf=dfold+f'features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'
    af=dfold+f'features/{pid}_OpenSMILE2.3.0_egemaps.csv'
    if not(os.path.exists(vf) and os.path.exists(af)):continue
    try:
        dv=pd.read_csv(vf,usecols=lambda c:c.strip() in set(['success']+AUS));dv.columns=[c.strip() for c in dv.columns]
        if 'success' in dv:dv=dv[dv['success']==1]
        AU=dv[AUS].values.astype(float)
        da=pd.read_csv(af,sep=';');da.columns=[c.strip() for c in da.columns]
        loud=da['Loudness_sma3'].values.astype(float)
    except:continue
    if len(AU)<30 or len(loud)<30:continue
    face=np.concatenate([[0],np.linalg.norm(np.diff(AU,axis=0),axis=1)])
    sc=seg_coupling(face,loud)
    if sc is not None and not np.isnan(sc).any():Seg.append(sc);Y.append(lab[pid])
report('E-DAIC',Seg,Y)
print('\n[비교] D-Vlog(exp115): 중반 40-80% 약화 유의. 재현 여부 판정.',flush=True)
print('DONE',flush=True)
