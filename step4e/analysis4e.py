import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
out=[]; P=lambda *a: out.append(' '.join(str(x) for x in a))
def wilson(k,n,z=1.959964):
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return c-h,c+h
def desc(x):
    x=pd.Series(x).dropna()
    if not len(x): return 'n=0'
    return f'median {x.median():.1f} [IQR {x.quantile(.25):.1f}, {x.quantile(.75):.1f}; range {x.min():.1f} to {x.max():.1f}] (n={len(x)})'
# paired(): d = easy - hard; returns (d, #d>tol [easy higher = hard lower], #|d|<=tol, #d<-tol [easy lower = hard higher]).
# fix: summary_thresholds.csv previously had hard_lower/hard_higher swapped, and summary_paths.csv
# called the easy-lower count "hard_lower_Ps". Labels corrected; no numbers changed.
def paired(e,h,col,tol=0.0):
    ix=e.index.intersection(h.index); d=(e[col][ix]-h[col][ix]).dropna()
    return d, int((d>tol).sum()), int((d.abs()<=tol).sum()), int((d< -tol).sum())
cond_lab=lambda s,r: 'easy' if s=='easy' else f'hard rel {r:.0f} ms'
# 1 nominal sweep
S=pd.read_csv('nominal_sweep.csv'); rows=[]
for (st,s,r),g in S.groupby(['stage','start','Trel_ms']):
    g=g.sort_values('Ps'); o=g.success.values.astype(int)
    rows.append(dict(stage=st,cond=cond_lab(s,r),lowest_Ps=g.Ps[g.success].min(),monotone=bool(np.all(np.diff(o)>=0))))
S1=pd.DataFrame(rows); S1.to_csv('summary_nominal_threshold.csv',index=False); P('## nominal steady-pressure sweep (5 Pa)'); P(S1.to_string())
# 2 thresholds per draw
T=pd.read_csv('thresholds.csv'); rows=[]
for st in 'BA':
    e=T[(T.stage==st)&(T.start=='easy')].set_index('draw')
    rows.append(dict(stage=st,cond='easy',threshold=desc(e.Ps_th)))
    for r in (5,20,100):
        h=T[(T.stage==st)&(T.start=='hard')&(T.Trel_ms.round()==r)].set_index('draw'); d,ep,eq,hl=paired(e,h,'Ps_th',tol=2.0)
        rows.append(dict(stage=st,cond=f'hard rel {r} ms',threshold=desc(h.Ps_th),easy_minus_hard=desc(d),hard_lower=ep,within_2Pa=eq,hard_higher=hl))
T1=pd.DataFrame(rows); T1.to_csv('summary_thresholds.csv',index=False); P('## steady-pressure thresholds, 20 parameter draws'); P(T1.to_string())
# 3 delays
D=pd.read_csv('delays.csv'); rows=[]
for (st,m),g in D.groupby(['stage','mult']):
    e=g[g.start=='easy'].set_index('draw')
    for (s,r),gg in g.groupby(['start','Trel_ms']):
        x=gg.set_index('draw'); k=int(x.success.sum()); n=len(x); lo,hi=wilson(k,n)
        row=dict(stage=st,mult=m,cond=cond_lab(s,r),success=f'{k}/{n} [{lo:.2f},{hi:.2f}]',delay_ms=desc(x.delay_ms[x.success]),kick_start=int((x.grew==0).sum()))
        if s=='hard':
            d,ep,eq,hl=paired(e[e.success],x[x.success],'delay_ms'); row.update(easy_minus_hard_ms=desc(d),hard_shorter=ep,hard_longer=hl)
            row['delay_from_release_end_ms']=desc(x.delay_from_release_end_ms[x.success])
        rows.append(row)
D1=pd.DataFrame(rows); D1.to_csv('summary_delays.csv',index=False); P('## onset delay at fixed pressure (50 draws)'); P(D1.to_string())
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for ax,st in zip(axs,'BA'):
    for (s,r),g in D[(D.stage==st)&D.success].groupby(['start','Trel_ms']):
        med=g.groupby('mult').delay_ms.median(); ax.plot(med.index,med.values,'o-',label=cond_lab(s,r))
    ax.set_yscale('log'); ax.set_xlabel('Ps / easy threshold (4c value)'); ax.set_ylabel('median delay to onset (ms)'); ax.set_title(f'Stage {st}: delay (easy: from pressure on; hard: from release start)'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('fig3_delays.png',dpi=130)
# 4 H4
H=pd.read_csv('h4.csv'); H['delay']=(H.t_on-0.005)*1e3
for st in 'BA':
    g=H[(H.stage==st)&H.success&(H.delay>0)]; b=np.polyfit(np.log(g.mult-1),np.log(g.delay),1)[0] if len(g)>3 else np.nan
    sl=[np.polyfit(np.log(x.mult-1),np.log(x.delay),1)[0] for _,x in g.groupby('seed') if x.mult.nunique()>2]
    P(f'## H4 stage {st}: pooled log-log slope {b:.2f}; per-seed slopes {desc(sl)}; successes {int(H[H.stage==st].success.sum())}/{(H.stage==st).sum()}; kick-start {int((g.grew==0).sum())}')
    P(g.groupby('mult').delay.median().round(1).to_string())
fig,ax=plt.subplots(figsize=(4.8,3.6))
for st in 'BA':
    g=H[(H.stage==st)&H.success&(H.delay>0)]; ax.loglog((g.mult-1),g.delay,'o',alpha=.5,label=f'Stage {st}')
ax.set_xlabel('Ps/Pth − 1'); ax.set_ylabel('onset delay (ms)'); ax.legend(); fig.tight_layout(); fig.savefig('fig5_h4.png',dpi=130)
# 5 paths
Pa=pd.read_csv('paths.csv'); rows=[]
for (st,m,Tr),g in Pa.groupby(['stage','mult','Tr_ms']):
    e=g[g.start=='easy'].set_index('draw'); ke=int(e.success.sum()); n=len(e); lo,hi=wilson(ke,n)
    rows.append(dict(stage=st,mult=m,Tr_ms=Tr,cond='easy',success=f'{ke}/{n} [{lo:.2f},{hi:.2f}]',Ps_on=desc(e.Ps_on[e.success]),t_after_ramp_ms=desc(e.t_after_ramp_ms[e.success])))
    for r,gg in g[g.start=='hard'].groupby('Trel_ms'):
        h=gg.set_index('draw'); kh=int(h.success.sum()); lo,hi=wilson(kh,len(h))
        es=e[e.success]; hs=h[h.success]; d,ep,eq,hl=paired(es,hs,'Ps_on',tol=0.5); dt,tp,tq,tl=paired(es,hs,'t_after_ramp_ms')
        bothfail=int(((~e.success)&(~h.success)).sum()); only_e=int((e.success&~h.success).sum()); only_h=int((~e.success&h.success).sum())
        rows.append(dict(stage=st,mult=m,Tr_ms=Tr,cond=f'hard rel {r:.0f} ms',success=f'{kh}/{len(h)} [{lo:.2f},{hi:.2f}]',Ps_on=desc(hs.Ps_on),t_after_ramp_ms=desc(hs.t_after_ramp_ms),
            easy_minus_hard_Ps=desc(d),easy_lower_Ps=f'{hl}/{len(d)} (equal {eq})',hard_earlier=f'{tp}/{len(dt)}',only_easy_starts=only_e,only_hard_starts=only_h,kick_start=int((hs.grew==0).sum()),
            max_gap_held_mm=f'{h.max_gap_while_held_mm.max():.3f}'))
P1=pd.DataFrame(rows); P1.to_csv('summary_paths.csv',index=False); P('## paths (H2/H3)'); P(P1.to_string())
fig,axs=plt.subplots(1,2,figsize=(11,3.8))
for ax,st in zip(axs,'BA'):
    g=Pa[(Pa.stage==st)&(Pa.mult==1.2)&Pa.success]
    for (s,r),gg in g.groupby(['start','Trel_ms']):
        med=gg.groupby('Tr_ms').t_after_ramp_ms.median(); ax.plot(med.index,med.values,'o-',label=cond_lab(s,r))
    ax.set_xscale('log'); ax.set_xlabel('pressure ramp T_r (ms)'); ax.set_ylabel('median onset time after ramp end (ms)'); ax.set_title(f'Stage {st}, 1.2×'); ax.axhline(0,color='k',lw=.5); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('fig4_paths.png',dpi=130)
# 6 map
if os.path.exists('map.csv'):
    M=pd.read_csv('map.csv')
    for st in 'BA':
        g=M[M.stage==st]; piv=g.pivot(index='x0_mm',columns='Q',values='Ps_th'); P(f'## map stage {st}'); P(piv.round(0).to_string())
        P('no onset <=2000:',int(g.Ps_th.isna().sum()),'/',len(g),' no onset at 800:',int(((g.Ps_th>800)|g.Ps_th.isna()).sum()))
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,st in zip(axs,'BA'):
        piv=M[M.stage==st].pivot(index='x0_mm',columns='Q',values='Ps_th'); im=ax.imshow(piv.values,aspect='auto',origin='lower',vmax=2000)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for (i,j),v in np.ndenumerate(piv.values): ax.text(j,i,'none' if np.isnan(v) else f'{v:.0f}',ha='center',va='center',fontsize=5,color='w' if (np.isnan(v) or v<1000) else 'k')
        ax.set_title(f'Stage {st} onset map (growth criterion)'); ax.set_xlabel('Q'); ax.set_ylabel('x0 (mm)'); plt.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig('fig6_maps.png',dpi=130)
fig,axs=plt.subplots(1,2,figsize=(10,3.6))
for ax,st in zip(axs,'BA'):
    data=[T[(T.stage==st)&(T.start=='easy')].Ps_th.dropna()]+[T[(T.stage==st)&(T.start=='hard')&(T.Trel_ms.round()==r)].Ps_th.dropna() for r in (5,20,100)]
    ax.boxplot(data,tick_labels=['easy','hard 5','hard 20','hard 100']); ax.set_ylabel('steady-pressure threshold (Pa)'); ax.set_title(f'Stage {st}, 20 parameter draws')
fig.tight_layout(); fig.savefig('fig2_thresholds.png',dpi=130)
open('analysis_output.txt','w').write('\n'.join(out)); print('\n'.join(out))
