import numpy as np, pandas as pd, time
from multiprocessing import Pool
from model import *
X0 = [-0.05,0,0.05,0.1,0.2,0.4,0.6,1.0]   # mm (coarse version of plan's 15 values)
QS = [0.5,0.8,1.0,1.4,2.0]                # plan's 8 values -> 5
def cell(c):
    x0mm,Q = c; v,n = threshold_bisect(0,2000,tol=8.0,Q=Q,x0=x0mm*1e-3)
    on8,rms8,f08 = onset_const(800,Q=Q,x0=x0mm*1e-3)
    lin = linear_threshold(Q, x0mm*1e-3)[1] if x0mm>0 else np.nan
    return dict(x0_mm=x0mm,Q=Q,Ps_th=v,no_onset_2000=np.isnan(v),runs=n,lin_Ps_th=lin,onset_at_800=on8,F0_800=f08)
if __name__=='__main__':
    t0=time.time()
    with Pool(8) as pool: r=pool.map(cell,[(x,q) for q in QS for x in X0])
    d=pd.DataFrame(r); d.to_csv('exp1_onset_map.csv',index=False); print(d.to_string()); print(time.time()-t0)
