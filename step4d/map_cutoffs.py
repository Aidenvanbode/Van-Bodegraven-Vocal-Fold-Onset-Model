import numpy as np, pandas as pd
S=pd.concat([pd.read_csv('../step4c/wide_sweep_full.csv'),pd.read_csv('../step4c/wide_sweep_refine10.csv')]).drop_duplicates(['x0_mm','Q','Ps','seed'])
S['acdc']=S.rms_cm3s/S.Udc_cm3s; rows=[]; cells=[]
for c in (0.0,0.01,0.02,0.05,0.10):
    S['ok']=S.on&(S.acdc>c)
    for (x,q),g in S.groupby(['x0_mm','Q']):
        fr=g.groupby('Ps').ok.mean(); ps=fr.index.values; maj=fr.values>=0.5
        lo=ps[maj].min() if maj.any() else np.nan
        cells.append(dict(cut=c,x0_mm=x,Q=q,lowest=lo,monotone=bool(np.all(np.diff(maj.astype(int))>=0))))
C=pd.DataFrame(cells); base=C[C.cut==0.02].set_index(['x0_mm','Q']).lowest
for c,g in C.groupby('cut'):
    g=g.set_index(['x0_mm','Q']); d=(g.lowest-base)
    rows.append(dict(cut=c,non_monotone=int((~g.monotone).sum()),no_onset_2000=int(g.lowest.isna().sum()),no_onset_800=int(((g.lowest>800)|g.lowest.isna()).sum()),
        cells_differing_from_2pct=int(((d.abs()>0)|(g.lowest.isna()!=base.isna())).sum()),median_shift_Pa=float(d.median()),max_abs_shift_Pa=float(d.abs().max())))
R=pd.DataFrame(rows); C.to_csv('map_cutoff_cells.csv',index=False); R.to_csv('map_cutoff_summary.csv',index=False); print(R.to_string())
# compare to Step 4 no-noise where available
m4=pd.read_csv('../step4/exp1_onset_map.csv')[['x0_mm','Q','Ps_th']]
for c in (0.0,0.01,0.02,0.05,0.10):
    j=C[C.cut==c].merge(m4,on=['x0_mm','Q']).dropna(); print(c,'n',len(j),'median |4c-step4|',np.median(np.abs(j.lowest-j.Ps_th)).round(1),'max',np.abs(j.lowest-j.Ps_th).max().round(1))
