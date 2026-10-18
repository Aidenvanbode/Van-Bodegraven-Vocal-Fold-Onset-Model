import numpy as np, pandas as pd, time, json
from multiprocessing import Pool
from model import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
t0=time.time()
# ---- 1a: hysteresis with slow ramps (raised-cosine up and down, 0->1000->0 Pa)
def hyst(Tr):
    p = params(); T = 2*Tr
    Pf = lambda t: 1000*np.sin(np.pi*t/T)**2
    t,y,U,Ps = simulate(p, Pf, T)
    tc, r = window_rms(U)
    above = r > 1e-6
    if not above.any(): return dict(Tr=Tr, Ps_on=np.nan, Ps_off=np.nan)
    i_on = np.argmax(above); i_off = len(above)-1-np.argmax(above[::-1])
    return dict(Tr=Tr, Ps_on=Pf(tc[i_on]), Ps_off=Pf(tc[min(i_off+1,len(tc)-1)]), ratio=Pf(tc[min(i_off+1,len(tc)-1)])/Pf(tc[i_on]))
# ---- 1b: why is threshold ~2x linear?  bisection under variants
def th(kw):
    name = kw.pop('name'); T = kw.pop('T', 1.0)
    v, n = threshold_bisect(tol=4.0, T=T, **kw); return dict(variant=name, Ps_th=v, runs=n)
variants = [dict(name='baseline'),
            dict(name='symmetric_forcing (no half-cycle switch)', flow_mode='symmetric'),
            dict(name='EPS=0.25um', EPS=0.25e-6), dict(name='EPS=4um', EPS=4e-6),
            dict(name='perturb 0.1um', y0=[1e-7,0,1e-7,0]), dict(name='perturb 20um', y0=[2e-5,0,2e-5,0]),
            dict(name='thr 0.1 cm3/s', thr=1e-7), dict(name='T=3 s', T=3.0),
            dict(name='rtol 1e-10', rtol=1e-10, atol=1e-13)]
if __name__=='__main__':
    with Pool(8) as pool:
        H_ = pool.map_async(hyst, [0.5, 2, 5, 10])
        va=[]
        for r in pool.imap_unordered(th, variants):
            print(r, round(time.time()-t0), flush=True); va.append(r)
        va=pd.DataFrame(va); hy = pd.DataFrame(H_.get())
    Gth, Pl = linear_threshold()
    hy.to_csv('part1_hysteresis.csv', index=False); va.to_csv('part1_threshold_variants.csv', index=False)
    print(hy.to_string()); print(va.to_string()); print('linear', Gth, Pl, 'elapsed', time.time()-t0)
