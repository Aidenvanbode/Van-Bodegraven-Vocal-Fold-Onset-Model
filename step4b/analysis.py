import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, linregress, norm
def proportion_confint(k,n,method='wilson',z=1.959964):
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return c-h, c+h
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*p[i]); adj[i]=min(1,run)
    return adj
out = []; P = lambda *a: out.append(' '.join(str(x) for x in a))
# Part 1
A = pd.read_csv('p1_threshold.csv'); Hd = pd.read_csv('p1_hyst_steps.csv'); R = pd.read_csv('p1_slow_ramps.csv')
P('## thresholds'); P(A.to_string())
for dP, g in Hd.groupby('dP'):
    up = g[g.dir=='up']; dn = g[g.dir=='down']; on = up[up.on].Ps.min(); off = dn[dn.on].Ps.min()
    P(f'hyst dP={dP}: onset {on}, lowest still-on going down {off}, ratio {off/on:.3f}')
P(R.to_string())
g = Hd[Hd.dP==5]; fig, ax = plt.subplots(figsize=(5,3.5))
for d, mk in (('up','o-'),('down','s--')): s = g[g.dir==d]; ax.semilogy(s.Ps, s.rms*1e6, mk, label=d)
ax.axhline(1, color='k', lw=.5); ax.set_xlabel('Ps (Pa)'); ax.set_ylabel('AC flow RMS (cm³/s)'); ax.legend(); ax.set_title('Hysteresis with 0.5% noise, no nudge'); fig.tight_layout(); fig.savefig('fig0_hysteresis_noise.png', dpi=130)
# Map
M = pd.read_csv('exp1_onset_map.csv'); piv = M.pivot(index='x0_mm', columns='Q', values='Ps_th')
P('## map (Ps_th, Pa)'); P(piv.round(0).to_string())
P('no onset <=2000:', int(M.Ps_th.isna().sum()), '/', len(M), ' no onset at 800:', int(((M.Ps_th>800)|M.Ps_th.isna()).sum()))
P('no-onset cells at 800:', M[(M.Ps_th>800)|M.Ps_th.isna()][['x0_mm','Q']].values.tolist())
fig, ax = plt.subplots(figsize=(6,4)); im = ax.imshow(piv.values, aspect='auto', origin='lower', cmap='viridis', vmax=2000)
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
for (i,j),v in np.ndenumerate(piv.values): ax.text(j,i,'none' if np.isnan(v) else f'{v:.0f}',ha='center',va='center',fontsize=5,color='w' if (np.isnan(v) or v<1000) else 'k')
ax.set_xlabel('Q'); ax.set_ylabel('x0 (mm)'); plt.colorbar(im,label='onset threshold Ps (Pa)'); ax.set_title('Onset map, 120 cells, 0.5% noise'); fig.tight_layout(); fig.savefig('fig1_onset_map.png', dpi=130)
# Paths
D = pd.read_csv('exp23_paths.csv'); rows=[]; pvals=[]
for (m,Tr), g in D.groupby(['mult','Tr_ms']):
    h = g[g.start=='hard'].set_index('draw'); e = g[g.start=='easy'].set_index('draw'); ix = h.index.intersection(e.index)
    for s, x in (('hard',h),('easy',e)):
        k=int(x.success.sum()); n=len(x); lo,hi=proportion_confint(k,n,method='wilson')
        rows.append(dict(mult=m,Tr_ms=Tr,start=s,n=n,success=k,p=k/n,wilson_lo=lo,wilson_hi=hi,med_Ps_onset=x.Ps_at_onset.median(),med_t_onset_ms=x.t_onset_ms.median(),med_pen_um=x.max_pen_um.median()))
    both = [i for i in ix if h.success[i] and e.success[i]]; dd = (e.Ps_at_onset[both]-h.Ps_at_onset[both]).dropna()
    if len(dd)>5 and (dd!=0).any():
        w = wilcoxon(dd); z = norm.isf(w.pvalue/2); r = z/np.sqrt(len(dd))
    else: w=None; r=np.nan
    rng=np.random.default_rng(0); se=e.success[ix].values.astype(float); sh=h.success[ix].values.astype(float)
    bs=[(se[b]-sh[b]).mean() for b in (rng.integers(0,len(ix),len(ix)) for _ in range(10000))]
    pvals.append(dict(mult=m,Tr_ms=Tr,n_pairs=len(dd),median_diff_easy_minus_hard_Pa=dd.median() if len(dd) else np.nan,
        wilcoxon_p=w.pvalue if w else np.nan, effect_r=r, succ_diff=se.mean()-sh.mean(), diff_ci_lo=np.percentile(bs,2.5), diff_ci_hi=np.percentile(bs,97.5)))
S = pd.DataFrame(rows); T = pd.DataFrame(pvals)
ok = T.wilcoxon_p.notna(); T.loc[ok,'p_holm'] = holm(T.wilcoxon_p[ok])
S.to_csv('exp23_summary.csv',index=False); T.to_csv('exp23_tests.csv',index=False)
P('## path summary'); P(S.round(3).to_string()); P('## tests'); P(T.round(5).to_string())
# ramp-speed factor (within start): paired fast vs slow
for (m,s), g in D.groupby(['mult','start']):
    f=g[g.Tr_ms==25].set_index('draw'); sl=g[g.Tr_ms==200].set_index('draw'); ix=[i for i in f.index.intersection(sl.index) if f.success[i] and sl.success[i]]
    dd=(sl.Ps_at_onset[ix]-f.Ps_at_onset[ix]).dropna()
    if len(dd)>5 and (dd!=0).any(): P(f'ramp factor {s} x{m}: slow-fast median {dd.median():.1f} Pa, n={len(dd)}, p={wilcoxon(dd).pvalue:.2g}')
fig, axs = plt.subplots(1,2,figsize=(9,3.5))
for ax,(col,lab) in zip(axs,(('p','success fraction'),('med_Ps_onset','median Ps at onset (Pa)'))):
    for (s,Tr),g in S.groupby(['start','Tr_ms']):
        y=g[col].values; ax.plot(g.mult, y, 'o-' if s=='hard' else 's--', label=f'{s}, {Tr:.0f} ms')
        if col=='p': ax.fill_between(g.mult, g.wilson_lo, g.wilson_hi, alpha=.15)
    ax.set_xlabel('target / threshold'); ax.set_ylabel(lab)
axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig('fig2_paths.png', dpi=130)
# H4
H = pd.read_csv('h4_const.csv').dropna(); H = H[H.t_onset_ms > 0]; H['dist']=(H.mult-1)*259.8; lr=linregress(np.log(H.dist),np.log(H.t_onset_ms))
ci = 1.96*lr.stderr; P(f'## H4 slope {lr.slope:.3f} 95%CI [{lr.slope-ci:.3f},{lr.slope+ci:.3f}] R2 {lr.rvalue**2:.3f} n {len(H)}'); P(H.groupby('mult').t_onset_ms.median().to_string())
fig, ax = plt.subplots(figsize=(4.5,3.5)); ax.loglog(H.dist,H.t_onset_ms,'o'); xx=np.linspace(H.dist.min(),H.dist.max()); ax.loglog(xx,np.exp(lr.intercept)*xx**lr.slope,'k-',label=f'slope {lr.slope:.2f}')
ax.set_xlabel('Ps − Ps,th (Pa)'); ax.set_ylabel('onset time (ms)'); ax.legend(); fig.tight_layout(); fig.savefig('fig4_h4.png', dpi=130)
open('analysis_output.txt','w').write('\n'.join(out)); print('\n'.join(out))
