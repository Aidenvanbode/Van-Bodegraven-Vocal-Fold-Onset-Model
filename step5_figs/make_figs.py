import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy import stats
F='../'; O='figs/'
EASY='#1f77b4'; HARD='#d62728'
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'savefig.dpi':300})
# ---------- Fig 1: schematic + paths
fig=plt.figure(figsize=(7.2,3.0)); ax=fig.add_axes([0.0,0.02,0.36,0.96]); ax.set_xlim(0,10); ax.set_ylim(-1.3,10); ax.axis('off')
for s in (-1,1):
    cx=5+s*1.3
    for (y,h,lab,w) in ((2.2,2.6,'m$_1$',1.3),(5.4,1.4,'m$_2$',1.0)):
        x=cx-(0 if s>0 else w); ax.add_patch(Rectangle((x,y),w,h,fc='#cccccc',ec='k'))
        ax.text(x+w/2,y+h/2,lab,ha='center',va='center')
        xs=cx+(w if s>0 else -w); xe=xs+s*1.6
        zz=np.linspace(xs,xe,40); ax.plot(zz,y+h/2+0.25*np.sin(np.linspace(0,8*np.pi,40)),'k',lw=.7)
        ax.plot([xe,xe],[y+h/2-0.5,y+h/2+0.5],'k',lw=1.5)
    ax.plot([cx+s*0.5]*2,[4.8,5.4],'k',lw=.7)
ax.text(5.0,4.9,'k$_c$',fontsize=7,ha='center')
ax.annotate('',xy=(5,1.6),xytext=(5,0.2),arrowprops=dict(arrowstyle='->'))
ax.text(5,0.0,'P$_s$ (subglottal)',ha='center',fontsize=8,va='top')
ax.annotate('',xy=(5,8.8),xytext=(5,7.2),arrowprops=dict(arrowstyle='->')); ax.text(5,9.0,'U$_g$ (glottal flow)',ha='center',fontsize=8)
ax.annotate('',xy=(3.7,1.8),xytext=(6.3,1.8),arrowprops=dict(arrowstyle='<->',lw=.7)); ax.text(5,1.75,'2x$_0$',ha='center',va='top',fontsize=7)
ax.text(0.2,9.6,'(a)',fontweight='bold')
t=np.linspace(0,0.5,1000)
for k,Tr in enumerate((0.025,0.2)):
    a1=fig.add_axes([0.45+k*0.29,0.58,0.24,0.33]); a2=fig.add_axes([0.45+k*0.29,0.14,0.24,0.33])
    P=np.minimum(t/Tr,1)*1.2; a1.plot(t*1e3,P,color='k'); a1.set_ylabel('P$_s$ / P$_{s,th}$' if k==0 else ''); a1.set_title(f'({"bc"[k]}) T$_r$ = {Tr*1e3:.0f} ms'); a1.set_xticklabels([])
    xh=np.where(t<Tr,-0.05,np.minimum(-0.05+(t-Tr)/1e-3*0.25,0.2)); a2.plot(t*1e3,xh,color=HARD,label='hard'); a2.plot(t*1e3,0.2+0*t,'--',color=EASY,label='easy')
    a2.set_xlabel('time (ms)'); a2.set_ylabel('x$_0$ (mm)' if k==0 else ''); a2.set_ylim(-0.1,0.3); a2.axhline(0,color='gray',lw=.5)
    if k==1: a2.legend(fontsize=7,loc='center right')
fig.savefig(O+'fig1_model_paths.png'); plt.close()
# ---------- Fig 2: onset map
M=pd.read_csv(F+'step4c/onset_map_4c.csv'); piv=M.pivot(index='x0_mm',columns='Q',values='Ps_th')
fig,ax=plt.subplots(figsize=(6.5,4.6)); im=ax.imshow(piv.values,aspect='auto',origin='lower',vmin=0,vmax=2000,cmap='viridis')
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'{q:g}' for q in piv.columns]); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f'{x:g}' for x in piv.index])
for (i,j),v in np.ndenumerate(piv.values):
    if np.isnan(v): ax.add_patch(Rectangle((j-.5,i-.5),1,1,fc='white',ec='gray',hatch='///')); ax.text(j,i,'none',ha='center',va='center',fontsize=6)
    else: ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=6,color='w' if v<1100 else 'k')
    if not np.isnan(v) and v>800: ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,ec='orange',lw=0.8))
ax.axhline(piv.index.get_loc(0.4)-0.5,color='r',lw=1.2)
ax.set_xlabel('stiffness factor Q (–)'); ax.set_ylabel('prephonatory half-gap x$_0$ (mm)'); plt.colorbar(im,label='onset threshold P$_{s,th}$ (Pa)')
fig.tight_layout(); fig.savefig(O+'fig3_onset_map.png'); plt.close()
# ---------- Fig 3: paths
S=pd.read_csv(F+'step4b/exp23_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
for st,c in (('easy',EASY),('hard',HARD)):
    for Tr,ls,mk in ((25,'-','o'),(200,'--','s')):
        g=S[(S.start==st)&(S.Tr_ms==Tr)].sort_values('mult'); off=(-0.006 if st=='easy' else 0.006)+(0 if Tr==25 else 0.003)
        axs[0].errorbar(g.mult+off,g.p,yerr=[g.p-g.wilson_lo,g.wilson_hi-g.p],color=c,ls=ls,marker=mk,ms=4,capsize=2,label=f'{st}, {Tr} ms')
        axs[1].plot(g.mult,g.med_Ps_onset,color=c,ls=ls,marker=mk,ms=4,label=f'{st}, {Tr} ms')
axs[0].set_xlabel('target pressure / P$_{s,th}$'); axs[0].set_ylabel('success fraction (95% Wilson CI)'); axs[0].set_title('(a) success'); axs[0].legend(fontsize=7)
axs[1].set_xlabel('target pressure / P$_{s,th}$'); axs[1].set_ylabel('median P$_s$ at onset (Pa)'); axs[1].set_title('(b) pressure at onset')
fig.tight_layout(); fig.savefig(O+'fig4_paths.png'); plt.close()
# ---------- Fig 4: robustness
T=pd.read_csv(F+'step4c/sens_tests.csv'); B=pd.read_csv(F+'step4c/stageB_paths_tests.csv')
lab={'baseline':'baseline\n(ε=1 µm, h=3k)','eps0.25um':'ε = 0.25 µm','eps4um':'ε = 4 µm','hc0.5':'h × 0.5','hc2':'h × 2','stageB':'Stage B\n(viscous)'}
D=pd.concat([T.rename(columns={'median_easy_minus_hard':'d'}),B.rename(columns={'median_easy_minus_hard':'d'})])
sets=['baseline','eps0.25um','eps4um','hc0.5','hc2','stageB']; x=np.arange(len(sets))
fig,ax=plt.subplots(figsize=(7.0,3.0))
for k,(Tr,c) in enumerate(((25,'#555555'),(200,'#bbbbbb'))):
    v=[D[(D.setting==s)&(D.Tr_ms==Tr)].d.values[0] for s in sets]; b=ax.bar(x+(k-.5)*0.38,v,0.38,color=c,ec='k',label=f'T$_r$ = {Tr} ms')
    for xi,vi in zip(x+(k-.5)*0.38,v): ax.text(xi,vi+3,f'+{vi:.0f}',ha='center',fontsize=7)
ax.axvline(4.5,color='k',ls=':',lw=.8); ax.set_xticks(x); ax.set_xticklabels([lab[s] for s in sets],fontsize=8)
ax.set_ylabel('median paired difference\nP$_s$ at onset, easy − hard (Pa)'); ax.set_ylim(0,200); ax.legend(fontsize=7)
ax.text(2,190,'Stage A, 1.2 × 259.8 Pa',ha='center',fontsize=7); ax.text(5,190,'1.2 × 195 Pa',ha='center',fontsize=7)
fig.tight_layout(); fig.savefig(O+'fig5_h2_robustness.png'); plt.close()
# ---------- Fig 5: H4
H=pd.read_csv(F+'step4b/h4_const.csv'); Hp=H[H.t_onset_ms>0].copy(); Hp['d']=(Hp.mult-1)*259.8
r=stats.linregress(np.log10(Hp.d),np.log10(Hp.t_onset_ms))
fig,ax=plt.subplots(figsize=(4.2,3.2)); ax.loglog(Hp.d,Hp.t_onset_ms,'o',color='k',ms=4,label='constant-P$_s$ runs (n = 19)')
xx=np.logspace(np.log10(Hp.d.min()),np.log10(Hp.d.max()),50); ax.loglog(xx,10**(r.intercept)*xx**r.slope,'k-',label=f'fit, slope {r.slope:.2f}')
ax.loglog(xx,10**(r.intercept)*xx.min()**r.slope*(xx/xx.min())**-1,'k:',label='slope −1 (linear Hopf)')
ax.set_xlabel('P$_s$ − P$_{s,th}$ (Pa)'); ax.set_ylabel('onset time (ms)'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(O+'fig6_h4.png'); plt.close()
print('H4 slope',r.slope,r.rvalue**2)
# ---------- Fig 6: gap curve
A=M[(M.Q==1.0)&(M.x0_mm<=0.6)].sort_values('x0_mm'); G=pd.read_csv(F+'step4c/stageB_gap_threshold_tol5.csv').groupby('x0_mm').Ps_th.agg(['mean','min','max']).reset_index()
fig,ax=plt.subplots(figsize=(4.6,3.2)); ax.plot(A.x0_mm,A.Ps_th,'o-',color='k',ms=4,label='Stage A (no viscous loss)')
ax.errorbar(G.x0_mm,G['mean'],yerr=[G['mean']-G['min'],G['max']-G['mean']],fmt='s--',color='#7f3fbf',ms=4,capsize=2,label='Stage B (viscous loss), 2 seeds')
ax.set_xlabel('prephonatory half-gap x$_0$ (mm)'); ax.set_ylabel('onset threshold P$_{s,th}$ (Pa)'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(O+'fig2_gap_curve.png'); plt.close()
print(A.to_string()); print(G.to_string())
