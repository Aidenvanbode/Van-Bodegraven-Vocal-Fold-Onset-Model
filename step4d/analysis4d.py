import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata
out=[]; P=lambda *a: out.append(' '.join(str(x) for x in a))
CUT=[0,1,2,5,10]
def wilson(k,n,z=1.959964):
    if n==0: return np.nan,np.nan
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return c-h,c+h
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); a=np.empty(m); r=0
    for i,j in enumerate(o): r=max(r,(m-i)*p[j]); a[j]=min(1,r)
    return a
def rbc(d):
    d=np.asarray(d); d=d[d!=0]; rk=rankdata(np.abs(d)); return (rk[d>0].sum()-rk[d<0].sum())/rk.sum() if len(d) else np.nan
def boot(a,b,n=10000):
    rng=np.random.default_rng(0); N=len(a); idx=rng.integers(0,N,(n,N)); d=a[idx].mean(1)-b[idx].mean(1); return np.percentile(d,[2.5,97.5])
D=pd.read_csv('path_runs.csv')
def h23(Dsub,label,cuts=CUT,metric='Ps_on'):
    rows=[]
    for c in cuts:
        for key,g in Dsub.groupby(['stage','mult','Tr_ms']):
            h=g[g.start=='hard'].set_index('draw'); e=g[g.start=='easy'].set_index('draw'); ix=h.index.intersection(e.index)
            sh=h[f'succ_{c}'][ix].values.astype(float); se=e[f'succ_{c}'][ix].values.astype(float)
            both=[i for i in ix if h[f'succ_{c}'][i] and e[f'succ_{c}'][i]]
            if metric=='Ps_on': dd=(e[f'Ps_on_{c}'][both]-h[f'Ps_on_{c}'][both]).dropna()
            else: dd=((e[f't_on_{c}'][both])-(h[f't_on_{c}'][both])).dropna()
            w=wilcoxon(dd).pvalue if len(dd)>5 and (dd!=0).any() else np.nan
            lo,hi=boot(se,sh); kh,ke=int(sh.sum()),int(se.sum()); n=len(ix)
            rows.append(dict(set=label,cut_pct=c,stage=key[0],mult=key[1],Tr_ms=key[2],n=n,succ_hard=kh/n,hard_lo=wilson(kh,n)[0],hard_hi=wilson(kh,n)[1],succ_easy=ke/n,easy_lo=wilson(ke,n)[0],easy_hi=wilson(ke,n)[1],
                succ_diff=(ke-kh)/n,diff_lo=lo,diff_hi=hi,n_pairs=len(dd),med_easy_minus_hard=dd.median() if len(dd) else np.nan,p=w,r_rb=rbc(dd)))
    T=pd.DataFrame(rows)
    for c in cuts:
        m=(T.cut_pct==c)&T.p.notna(); T.loc[m,'p_holm']=holm(T.p[m])
    return T
M=D[D.exp=='main']; T=h23(M,'main'); T.to_csv('h23_by_cutoff.csv',index=False)
P('## H2/H3 by AC/DC cut-off (Ps at onset, easy-hard)'); P(T[['cut_pct','stage','mult','Tr_ms','succ_hard','succ_easy','diff_lo','diff_hi','n_pairs','med_easy_minus_hard','p_holm','r_rb']].round(4).to_string())
Tt=h23(M,'main_time',cuts=[2],metric='t_on'); Tt.to_csv('h2_onset_time_2pct.csv',index=False)
P('## onset time easy-hard (ms, from path start; 2% rule)'); P(Tt[['stage','mult','Tr_ms','n_pairs','med_easy_minus_hard','p_holm','r_rb']].round(4).to_string())
# phase of hard onset
H=D[(D.start=='hard')]
ph=H.groupby(['exp','stage','Tr_ms','rel_ms','eps_um','mult']).agg(n=('draw','size'),blown_open=('blown_open_while_pressed','mean'),Ps_first_open=('Ps_first_open','median'),
    pressed=('phase',lambda s:(s=='pressed').mean()),release=('phase',lambda s:(s=='release').mean()),after=('phase',lambda s:(s=='after').mean()),
    t_on_ms=('t_on_2','median'),Ps_on=('Ps_on_2','median'),gap_on_mm=('gap_on_mm','median'),amp_on=('amp_on_cm3s','median')).reset_index()
E=D[D.start=='easy'].groupby(['exp','stage','Tr_ms','mult']).agg(easy_t_on_ms=('t_on_2','median'),easy_Ps_on=('Ps_on_2','median'),easy_amp_on=('amp_on_cm3s','median')).reset_index()
ph=ph.merge(E.drop(columns='exp'),on=['stage','Tr_ms','mult'],how='left'); ph=ph.drop_duplicates(); ph.to_csv('hard_onset_phase.csv',index=False)
P('## hard-onset phase (2% rule)'); P(ph.round(3).to_string())
# release-duration & eps tests (2% rule)
R=[]
for (ex,st,Tr,rel,eps),g in D[(D.start=='hard')&(D.mult==1.2)].groupby(['exp','stage','Tr_ms','rel_ms','eps_um']):
    e=D[(D.start=='easy')&(D.mult==1.2)&(D.stage==st)&(D.Tr_ms==Tr)&(D.exp==('eps0.05' if ex=='eps0.05' else 'main'))].set_index('draw'); h=g.set_index('draw'); ix=h.index.intersection(e.index)
    both=[i for i in ix if h.succ_2[i] and e.succ_2[i]]; dd=(e.Ps_on_2[both]-h.Ps_on_2[both]).dropna(); dt=(e.t_on_2[both]-h.t_on_2[both]).dropna()
    R.append(dict(exp=ex,stage=st,Tr_ms=Tr,rel_ms=rel,eps_um=eps,n_pairs=len(dd),med_dPs=dd.median(),p_Ps=wilcoxon(dd).pvalue if (dd!=0).any() else np.nan,r_Ps=rbc(dd),med_dt_ms=dt.median(),p_t=wilcoxon(dt).pvalue if (dt!=0).any() else np.nan,succ_hard=h.succ_2[ix].mean(),succ_easy=e.succ_2[ix].mean()))
R=pd.DataFrame(R); R['p_Ps_holm']=holm(R.p_Ps.fillna(1)); R.to_csv('release_eps_tests.csv',index=False); P('## release duration / smoothing (easy-hard, 2% rule)'); P(R.round(5).to_string())
# kick
K=pd.read_csv('kick_runs.csv'); K['cond']=K.start+'_rel'+K.rel_ms.round().astype(int).astype(str)
KS=[]
for (st,m,cond),g in K.groupby(['stage','mult','cond']):
    k=int(g.succ_2.sum()); n=len(g); lo,hi=wilson(k,n)
    KS.append(dict(stage=st,mult=m,cond=cond,n=n,success=k/n,lo=lo,hi=hi,med_t_after_T0_ms=g.t_after_T0_2.median(),med_amp_on=g.amp_on_cm3s.median()))
KS=pd.DataFrame(KS); KS.to_csv('kick_summary.csv',index=False); P('## kick test (constant Ps, release at 150 ms)'); P(KS.round(3).to_string())
fig,axs=plt.subplots(1,2,figsize=(10,3.6))
for ax,st in zip(axs,'AB'):
    for cond,g in KS[KS.stage==st].groupby('cond'): ax.plot(g.mult,g.med_t_after_T0_ms,'o-',label=cond)
    ax.set_xlabel('Ps / static threshold (0.2 mm)'); ax.set_ylabel('median onset time after t=150 ms (ms)'); ax.set_title(f'Stage {st}: release vs. no release'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('fig2_kick.png',dpi=130)
# cutoff sensitivity figure
fig,axs=plt.subplots(1,2,figsize=(10,3.6))
for (st,m,Tr),g in T.groupby(['stage','mult','Tr_ms']):
    axs[0].plot(g.cut_pct,g.med_easy_minus_hard,'o-',label=f'{st} {m}x {Tr:.0f}ms'); axs[1].plot(g.cut_pct,g.succ_easy-g.succ_hard,'o-')
axs[0].set_xlabel('AC/DC cut-off (%)'); axs[0].set_ylabel('median Ps_onset easy − hard (Pa)'); axs[0].axhline(0,color='k'); axs[0].legend(fontsize=6)
axs[1].set_xlabel('AC/DC cut-off (%)'); axs[1].set_ylabel('success easy − hard'); axs[1].axhline(0,color='k'); fig.tight_layout(); fig.savefig('fig3_cutoff_paths.png',dpi=130)
fig,ax=plt.subplots(figsize=(6,3.6)); sub=R[R.exp!='eps0.05']; lab=[]
for k,(_,r) in enumerate(R.iterrows()): ax.bar(k,r.med_dPs,color={'main':'C0','release':'C1','eps0.05':'C2'}[r.exp]); lab.append(f'{r.stage} {r.Tr_ms:.0f}ms rel{r.rel_ms:.0f} eps{r.eps_um:g}')
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab,rotation=60,fontsize=6); ax.axhline(0,color='k'); ax.set_ylabel('median Ps_onset easy − hard (Pa)'); fig.tight_layout(); fig.savefig('fig4_release_eps.png',dpi=130)
open('analysis_output.txt','w').write('\n'.join(out)); print('\n'.join(out))
