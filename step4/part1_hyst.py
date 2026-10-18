"""Quasi-static hysteresis: step Ps up then down, hold each level, carry state over,
re-seed a 0.1 um perturbation at each step (stand-in for physiological noise).
Reason: with a smooth noiseless ramp, the 1 um seed decays to ~0 below threshold and
onset is delayed far past the Hopf point (dynamic bifurcation delay) -> ratio artefact."""
import numpy as np, pandas as pd, time
from multiprocessing import Pool
from model import *
def run(cfg):
    dP, hold = cfg; p = params(); y = [1e-7,0,1e-7,0]; rows=[]
    levels = list(np.arange(150, 451, dP)) ; levels = levels + levels[::-1][1:]
    for k,Ps in enumerate(levels):
        y = list(y); y[0] += 1e-7; y[2] += 1e-7
        t,Y,U,_ = simulate(p, lambda t,Ps=Ps: Ps, hold, y0=y)
        on, rms = classify(t, U, t_keep=min(0.1,hold/3))
        rows.append(dict(dP=dP, hold=hold, dir='up' if k < len(levels)//2+1 else 'down', Ps=Ps, on=on, rms=rms)); y = Y[:,-1]
    d = pd.DataFrame(rows); up=d[d.dir=='up']; dn=d[d.dir=='down']
    Pon = up[up.on].Ps.min(); Poff = dn[~dn.on].Ps.max() if (~dn.on).any() else np.nan
    return d, dict(dP=dP, hold=hold, Ps_on=Pon, Ps_off_last_off=Poff, Ps_off_lowest_on=dn[dn.on].Ps.min())
if __name__=='__main__':
    t0=time.time()
    with Pool(4) as pool: res = pool.map(run, [(20,0.3),(10,0.3),(10,0.6),(5,0.6)])
    pd.concat([r[0] for r in res]).to_csv('part1_hyst_steps.csv', index=False)
    s=pd.DataFrame([r[1] for r in res]); s['ratio']=s.Ps_off_lowest_on/s.Ps_on
    s.to_csv('part1_hysteresis_summary.csv', index=False); print(s.to_string(), time.time()-t0)
