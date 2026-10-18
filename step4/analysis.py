import numpy as np, pandas as pd
from scipy import stats
from multiprocessing import Pool
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from model import *
def wilson(k,n,z=1.96):
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return c-h,c+h
def h4(dP):
    Pth=259.77; p=params(); t,y,U,_=simulate(p,lambda t:(Pth+dP)*min(1,t/0.005),2.0)
    tc,r=window_rms(U); a=r>1e-6
    if not a[-1]: return dict(dP=dP,t_on_ms=np.nan)
    k=len(a)-1
    while k>0 and a[k-1]: k-=1
    return dict(dP=dP,t_on_ms=(tc[k]-0.005)*1e3)
if __name__=='__main__':
    out=[]
    # ---- Exp1 figure
    e1=pd.read_csv('exp1_onset_map.csv'); piv=e1.pivot(index='Q',columns='x0_mm',values='Ps_th')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    im=ax[0].imshow(piv.values,origin='lower',aspect='auto',cmap='viridis')
    ax[0].set_xticks(range(len(piv.columns)),piv.columns); ax[0].set_yticks(range(len(piv.index)),piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v=piv.values[i,j]; ax[0].text(j,i,'none\n<2000' if np.isnan(v) else f'{v:.0f}',ha='center',va='center',fontsize=7,color='w')
    ax[0].set_xlabel('prephonatory half-gap x0 (mm)'); ax[0].set_ylabel('stiffness factor Q'); ax[0].set_title('Onset threshold pressure Ps_th (Pa), Stage A')
    plt.colorbar(im,ax=ax[0])
    for Q,g in e1.groupby('Q'):
        l,=ax[1].plot(g.x0_mm,g.Ps_th,'o-',label=f'Q={Q} sim'); ax[1].plot(g.x0_mm,g.lin_Ps_th,'--',color=l.get_color(),alpha=.6)
    ax[1].axhline(800,color='k',lw=.8,ls=':'); ax[1].set_xlabel('x0 (mm)'); ax[1].set_ylabel('Ps_th (Pa)'); ax[1].set_title('Threshold vs gap (dashed = linear Hopf theory)'); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig('fig1_onset_map.png',dpi=130); plt.close()
    g=e1[e1.x0_mm>0]; ratio=(g.Ps_th/g.lin_Ps_th); out.append(f'Exp1 sim/linear ratio (x0>0, finite): mean {ratio.mean():.3f}, range {ratio.min():.3f}-{ratio.max():.3f}')
    out.append(f'Exp1 cells with no onset <=2000 Pa: {int(e1.no_onset_2000.sum())}/{len(e1)}; cells NOT oscillating at 800 Pa: {int((~e1.onset_at_800).sum())}/{len(e1)}')
    for Q,gq in e1.groupby('Q'):
        ok=gq[gq.onset_at_800].x0_mm; out.append(f'  Q={Q}: onset at 800 Pa for x0 in {sorted(ok.tolist())} mm')
    # ---- Exp2
    e2=pd.read_csv('exp2_paths.csv')
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    for k,Q in enumerate([0.8,1.0,1.4]):
        for pa,m in zip(['H0','H1','E1','E2'],['s','^','o','D']):
            g=e2[(e2.Q==Q)&(e2.path==pa)]; ax[k].plot(g.Tr_ms,g.t_onset_ms,m+'-',label=pa)
        ax[k].plot([1,400],[1,400],'k:',lw=.7,label='onset = T_r'); ax[k].set_xscale('log'); ax[k].set_title(f'Q={Q}'); ax[k].set_xlabel('ramp duration T_r (ms)')
    ax[0].set_ylabel('onset time (ms, 10-ms resolution)'); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig('fig2_onset_times.png',dpi=130); plt.close()
    w=e2.pivot_table(index=['Tr_ms','Q'],columns='path',values='Ps_at_onset')
    easy=w[['E1','E2']].mean(axis=1); hard=w[['H0','H1']].mean(axis=1); m=easy.notna()&hard.notna()
    st=stats.wilcoxon(easy[m],hard[m]); out.append(f'H2 Wilcoxon (all T_r, n={m.sum()} pairs): median(easy-hard)={np.median(easy[m]-hard[m]):.0f} Pa, W={st.statistic:.1f}, p={st.pvalue:.2g}')
    m2=m&(w.index.get_level_values(0)>=50)
    st2=stats.wilcoxon(easy[m2],hard[m2]); out.append(f'H2 Wilcoxon (T_r>=50 ms, n={m2.sum()}): median(easy-hard)={np.median(easy[m2]-hard[m2]):.0f} Pa, p={st2.pvalue:.2g}')
    out.append(f'Exp2 success: {int(e2.success.sum())}/{len(e2)}; failures: '+str(e2[~e2.success][["path","Tr_ms","Q"]].values.tolist()))
    # ---- Exp3
    e3=pd.read_csv('exp3_robustness.csv'); rows=[]
    for (pa,Tr),g in e3.groupby(['path','Tr_ms']):
        k=int(g.success.sum()); n=len(g); lo,hi=wilson(k,n); rows.append(dict(path=pa,Tr_ms=Tr,k=k,n=n,p=k/n,lo=lo,hi=hi))
    R=pd.DataFrame(rows); R.to_csv('exp3_summary.csv',index=False); out.append('Exp3 success fractions:\n'+R.to_string(index=False))
    rng=np.random.default_rng(0); B=10000
    for Tr in [50.0,200.0]:
        se=e3[(e3.Tr_ms==Tr)&e3.path.isin(['E1','E2'])].success.values; sh=e3[(e3.Tr_ms==Tr)&e3.path.isin(['H0','H1'])].success.values
        d=[rng.choice(se,len(se)).mean()-rng.choice(sh,len(sh)).mean() for _ in range(B)]
        out.append(f'H3 T_r={Tr:.0f}: p_easy-p_hard={se.mean()-sh.mean():.3f}, bootstrap 95% CI [{np.percentile(d,2.5):.3f},{np.percentile(d,97.5):.3f}]')
    fig,ax=plt.subplots(figsize=(6,3.5)); lab=[f'{a} {b:.0f}ms' for a,b in zip(R.path,R.Tr_ms)]
    ax.errorbar(range(len(R)),R.p,yerr=[(R.p-R.lo).clip(0),(R.hi-R.p).clip(0)],fmt='o',capsize=4); ax.set_xticks(range(len(R)),lab,rotation=45); ax.set_ylim(0.9,1.01); ax.set_ylabel('success fraction (95% Wilson CI)'); ax.set_title(f'Exp3 robustness, N={R.n.iloc[0]} draws/condition, ±10% params')
    plt.tight_layout(); plt.savefig('fig3_robustness.png',dpi=130); plt.close()
    # ---- H4
    with Pool(8) as pool: h=pd.DataFrame(pool.map(h4,[5,10,20,40,80,160,320]))
    h.to_csv('h4_onset_time_vs_distance.csv',index=False); hh=h.dropna(); hh=hh[hh.t_on_ms>0]
    r=stats.linregress(np.log(hh.dP),np.log(hh.t_on_ms)); ci=1.96*r.stderr
    out.append('H4 onset time vs (Ps-Pth):\n'+h.to_string(index=False)+f'\n  log-log slope {r.slope:.2f} (95% CI {r.slope-ci:.2f} to {r.slope+ci:.2f}), R^2={r.rvalue**2:.2f}')
    fig,ax=plt.subplots(figsize=(4.5,3.5)); ax.loglog(hh.dP,hh.t_on_ms,'o-'); ax.set_xlabel('Ps - Ps_th (Pa)'); ax.set_ylabel('onset time (ms)'); ax.set_title(f'H4: slope {r.slope:.2f}'); plt.tight_layout(); plt.savefig('fig4_h4.png',dpi=130); plt.close()
    # Part1 figure
    s=pd.read_csv('part1_hyst_steps.csv'); s=s[(s.dP==5)]
    fig,ax=plt.subplots(figsize=(5,3.5))
    for dr,mk in [('up','^-'),('down','v-')]: g=s[s.dir==dr]; ax.plot(g.Ps,g.rms*1e6,mk,label=dr)
    ax.set_xlabel('Ps (Pa)'); ax.set_ylabel('AC flow RMS (cm³/s)'); ax.set_title('Quasi-static hysteresis (5 Pa steps, 0.6 s hold)'); ax.legend(); plt.tight_layout(); plt.savefig('fig0_hysteresis.png',dpi=130); plt.close()
    open('analysis_output.txt','w').write('\n'.join(out)); print('\n'.join(out))
