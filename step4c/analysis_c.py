import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, norm, mannwhitneyu
out=[]; P=lambda *a: out.append(' '.join(str(x) for x in a))
def wilson(k,n,z=1.959964):
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return c-h,c+h
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); a=np.empty(m); r=0
    for i,j in enumerate(o): r=max(r,(m-i)*p[j]); a[j]=min(1,r)
    return a
# 1. wide sweep
S=pd.concat([pd.read_csv('wide_sweep_full.csv'),pd.read_csv('wide_sweep_refine10.csv')]).drop_duplicates(['x0_mm','Q','Ps','seed']); rows=[]
# noise-floor fix: the 1 cm3/s absolute criterion is crossed by noise-driven flow at wide gaps; also require AC/DC > 2 %
S['acdc']=S.rms_cm3s/S.Udc_cm3s; S['on_raw']=S.on; S['on']=S.on_raw&(S.acdc>0.02); S.to_csv('wide_sweep_reclassified.csv',index=False)
P('runs flipped by AC/DC criterion:',int((S.on_raw!=S.on).sum()),'of',len(S))
for (x,q),g in S.groupby(['x0_mm','Q']):
    frac=g.groupby('Ps').on.mean(); raw=g.groupby('Ps').on_raw.mean(); ps=frac.index.values; maj=frac.values>=0.5
    lo=ps[maj].min() if maj.any() else np.nan
    mono=bool(np.all(np.diff(maj.astype(int))>=0)); gaps=[int(p) for p in ps[(~maj)&(ps>lo)]] if maj.any() else []
    rows.append(dict(x0_mm=x,Q=q,lowest_raw=ps[raw.values>=0.5].min() if (raw.values>=0.5).any() else np.nan,lowest_onset=lo,monotone=mono,
        nononset_above_lowest=gaps,mixed_seed_levels=int(((frac>0)&(frac<1)).sum()),raw_monotone=bool(np.all(np.diff((raw.values>=0.5).astype(int))>=0))))
W=pd.DataFrame(rows)
m4=pd.read_csv('../step4/exp1_onset_map.csv'); m4b=pd.read_csv('../step4b/exp1_onset_map.csv')
c4=[c for c in m4.columns if 'th' in c.lower()][0]
W=W.merge(m4b.rename(columns={'Ps_th':'step4b_bisect'})[['x0_mm','Q','step4b_bisect']],on=['x0_mm','Q'],how='left')
m4['x0_mm']=m4[[c for c in m4.columns if 'x0' in c][0]]*(1e3 if m4[[c for c in m4.columns if 'x0' in c][0]].max()<0.01 else 1)
W=W.merge(m4[['x0_mm','Q',c4]].rename(columns={c4:'step4_nonoise'}),on=['x0_mm','Q'],how='left')
W.to_csv('wide_sweep_summary.csv',index=False); P('## wide sweep'); P(W.to_string())
P('non-monotone cells (fixed criterion):',int((~W.monotone).sum()),'/',len(W),' raw criterion:',int((~W.raw_monotone).sum()),' cells with mixed-seed levels:',int((W.mixed_seed_levels>0).sum()))
# full map = 4b bisection for x0<0.4, sweep for >=0.4
M=m4b[m4b.x0_mm<0.4][['x0_mm','Q','Ps_th']].copy(); M=pd.concat([M,W[['x0_mm','Q','lowest_onset']].rename(columns={'lowest_onset':'Ps_th'})]); M.to_csv('onset_map_4c.csv',index=False)
piv=M.pivot(index='x0_mm',columns='Q',values='Ps_th')
fig,ax=plt.subplots(figsize=(6.5,4.5)); im=ax.imshow(piv.values,aspect='auto',origin='lower',vmax=2000)
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
for (i,j),v in np.ndenumerate(piv.values): ax.text(j,i,'none' if np.isnan(v) else f'{v:.0f}',ha='center',va='center',fontsize=5,color='w' if (np.isnan(v) or v<1000) else 'k')
ax.axhline(piv.index.get_loc(0.4)-0.5,color='r'); ax.set_xlabel('Q'); ax.set_ylabel('x0 (mm)'); plt.colorbar(im,label='lowest onset Ps (Pa)')
ax.set_title('Stage A onset map: bisection (x0<0.4) / direct sweep (x0>=0.4)'); fig.tight_layout(); fig.savefig('fig1_onset_map_4c.png',dpi=130)
P('no onset <=2000:',int(M.Ps_th.isna().sum()),' no onset at 800:',int(((M.Ps_th>800)|M.Ps_th.isna()).sum()),'/',len(M))
# example sweep plots
fig,axs=plt.subplots(1,2,figsize=(9,3.5))
for ax,q in zip(axs,(1.0,1.7)):
    for x in (0.4,0.6,0.8,1.0):
        g=S[(S.Q==q)&(S.x0_mm==x)&(S.seed==1)].sort_values('Ps'); ax.semilogy(g.Ps,g.rms_cm3s,'.-',label=f'x0={x}')
    ax.axhline(1,color='k',lw=.5); ax.set_title(f'Q={q}, seed 1'); ax.set_xlabel('Ps (Pa)'); ax.set_ylabel('AC flow RMS (cm³/s)')
axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig('fig2_wide_sweeps.png',dpi=130)
# 2. Stage B
G=pd.read_csv('stageB_gap_threshold_tol5.csv'); P('## Stage B gap curve'); P(G.to_string())
gB=G.groupby('x0_mm',as_index=False).Ps_th.mean().sort_values('x0_mm'); gA=m4b[m4b.Q==1.0].sort_values('x0_mm')
fig,ax=plt.subplots(figsize=(5,3.5)); ax.plot(gA.x0_mm[gA.x0_mm<=0.6],gA.Ps_th[gA.x0_mm<=0.6],'o-',label='Stage A (4b)'); ax.plot(gB.x0_mm,gB.Ps_th,'s-',label='Stage B (4c)')
ax.set_xlabel('x0 (mm)'); ax.set_ylabel('threshold Ps (Pa)'); ax.legend(); ax.set_title('Threshold vs gap, Q=1'); fig.tight_layout(); fig.savefig('fig3_gap_curve.png',dpi=130)
i=gB.Ps_th.idxmin(); P('Stage B min at x0 =',gB.x0_mm[i],'Ps',gB.Ps_th[i])
Hy=pd.read_csv('stageB_hyst.csv'); up=Hy[Hy.dir=='up']; dn=Hy[Hy.dir=='down']; on=up[up.on].Ps.min(); off=dn[dn.on].Ps.min(); P('Stage B hyst onset',on,'lowest-on down',off,'ratio',off/on)
# 3. paths
from scipy.stats import rankdata
def rbc(d):
    d=np.asarray(d); d=d[d!=0]; rk=rankdata(np.abs(d)); return (rk[d>0].sum()-rk[d<0].sum())/rk.sum()
def comp(D,label):
    rows=[]; tests=[]
    for (st,Tr),g in D.groupby(['setting','Tr_ms']):
        h=g[g.start=='hard'].set_index('draw'); e=g[g.start=='easy'].set_index('draw'); ix=h.index.intersection(e.index)
        for s,x in (('hard',h),('easy',e)):
            k=int(x.success.sum()); n=len(x); lo,hi=wilson(k,n); rows.append(dict(setting=st,Tr_ms=Tr,start=s,n=n,success=k,p=k/n,wilson_lo=lo,wilson_hi=hi,med_Ps_onset=x.Ps_at_onset.median()))
        b=[i for i in ix if h.success[i] and e.success[i]]; dd=(e.Ps_at_onset[b]-h.Ps_at_onset[b]).dropna()
        w=wilcoxon(dd) if len(dd)>5 and (dd!=0).any() else None
        rng=np.random.default_rng(0); se=e.success[ix].values.astype(float); sh=h.success[ix].values.astype(float)
        bs=[(se[j]-sh[j]).mean() for j in (rng.integers(0,len(ix),len(ix)) for _ in range(10000))]
        tests.append(dict(setting=st,Tr_ms=Tr,n_pairs=len(dd),median_easy_minus_hard=dd.median(),p=w.pvalue if w else np.nan,r=rbc(dd) if w else np.nan,
            succ_diff=se.mean()-sh.mean(),diff_lo=np.percentile(bs,2.5),diff_hi=np.percentile(bs,97.5)))
    R=pd.DataFrame(rows); T=pd.DataFrame(tests); ok=T.p.notna(); T.loc[ok,'p_holm']=holm(T.p[ok]); P(f'## {label}'); P(R.round(3).to_string()); P(T.round(5).to_string()); return R,T
BR,BT=comp(pd.concat([pd.read_csv('stageB_paths.csv'),pd.read_csv('stageB_paths_100_199.csv')]),'Stage B paths 1.2x'); BR.to_csv('stageB_paths_summary.csv',index=False); BT.to_csv('stageB_paths_tests.csv',index=False)
b4=pd.read_csv('../step4b/exp23_paths.csv'); b4=b4[(b4.mult==1.2)&(b4.Tr_ms==25)].assign(setting='baseline',stage='A')
SR,ST=comp(pd.concat([pd.read_csv('sens_paths.csv'),pd.read_csv('sens_paths_extra.csv'),b4[pd.read_csv('sens_paths.csv').columns.intersection(b4.columns)]]),'Contact sensitivity (Stage A, 1.2x)'); SR.to_csv('sens_summary.csv',index=False); ST.to_csv('sens_tests.csv',index=False)
P('## pressed thresholds'); P(pd.read_csv('sens_pressed_threshold.csv').to_string())
fig,ax=plt.subplots(figsize=(6,3.5)); lab=[]; 
for k,(_,r) in enumerate(ST.iterrows()): ax.bar(k,r.median_easy_minus_hard); lab.append(f'{r.setting} {r.Tr_ms:.0f}ms')
k0=len(lab)
for _,r in BT.iterrows(): ax.bar(k0,r.median_easy_minus_hard,color='C1'); lab.append(f'Stage B {r.Tr_ms:.0f}ms'); k0+=1
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab,rotation=30,fontsize=7); ax.axhline(0,color='k'); ax.set_ylabel('median Ps_onset easy − hard (Pa)'); fig.tight_layout(); fig.savefig('fig4_h2_robustness.png',dpi=130)
open('analysis_output.txt','w').write('\n'.join(out)); print('\n'.join(out))
