import time, numpy as np, pandas as pd
from multiprocessing import Pool
from model_e import *
XT = 0.2e-3; PTH = {'A': 259.8, 'B': 195.3125}; T_HOLD = 0.10
def draw(i):
    if i < 0: return np.ones(7)
    return np.random.default_rng(1000 + i).uniform(0.9, 1.1, 7)
def setup(stage, i, **extra):
    f = draw(i); kw = {k: BASE[k] * fi for k, fi in zip(['k1','k2','kc','m1','m2'], f[:5])}; kw.update(extra)
    return params(stage=stage, **kw), XT * f[5], f[6]
def const_run(stage, start, Ps, i, Trel, seed, Tafter=1.0):
    """constant Ps (5 ms rise). easy: open from t=0; hard: held until T_HOLD then released over Trel."""
    p, x0, _ = setup(stage, i); Pf = lambda t: Ps * min(t / 0.005, 1)
    if start == 'hard': w = w_release(T_HOLD, Trel); T = T_HOLD + Trel + Tafter; t0 = T_HOLD
    else: w = None; T = Tafter + 0.1; t0 = 0.005
    t, y, U, _ = simulate(p, Pf, T, x0, w, seed=seed); o = onset_growth(t, U, Pf, t_from=0.0); o['delay_ms'] = (o['t_on'] - t0) * 1e3
    if start == 'hard': o['delay_from_release_end_ms'] = (o['t_on'] - T_HOLD - Trel) * 1e3
    return o
def sweep_nominal(c):
    stage, start, Trel, Ps = c; o = const_run(stage, start, Ps, -1, Trel, 7)
    return dict(stage=stage, start=start, Trel_ms=Trel*1e3, Ps=Ps, **o)
def thr(c):
    stage, start, Trel, i = c; lo, hi = 0.5 * PTH[stage], 1.6 * PTH[stage]; n = 0
    if not const_run(stage, start, hi, i, Trel, 7 + i)['success']: return dict(stage=stage, start=start, Trel_ms=Trel*1e3, draw=i, Ps_th=np.nan)
    while hi - lo > 2.0:
        mid = 0.5 * (lo + hi)
        if const_run(stage, start, mid, i, Trel, 7 + i)['success']: hi = mid
        else: lo = mid
    return dict(stage=stage, start=start, Trel_ms=Trel*1e3, draw=i, Ps_th=0.5 * (lo + hi))
def delay(c):
    stage, start, Trel, m, i = c; o = const_run(stage, start, m * PTH[stage], i, Trel, 3000 + i)
    return dict(stage=stage, start=start, Trel_ms=Trel*1e3, mult=m, draw=i, **o)
def path(c):
    stage, start, Tr, Trel, m, i = c; p, x0, fP = setup(stage, i); Pt = m * PTH[stage] * fP
    Pf = lambda t: Pt * min(t / Tr, 1); w = w_release(Tr, Trel) if start == 'hard' else None
    T = Tr + Trel + 1.0; t, y, U, _ = simulate(p, Pf, T, x0, w, seed=5000 + i); o = onset_growth(t, U, Pf)
    held_max = float((x0 + np.minimum(y[0], y[2]))[t < Tr].max() * 1e3) if start == 'hard' else np.nan
    return dict(stage=stage, start=start, Tr_ms=Tr*1e3, Trel_ms=Trel*1e3, mult=m, draw=i, Ps_target=Pt,
                t_after_ramp_ms=(o['t_on'] - Tr) * 1e3, max_gap_while_held_mm=held_max, **o)
def cell(c):
    stage, x0, Q = c; lo, hi = 0.0, 2000.0
    def on(Ps):
        p = params(stage=stage, x0=x0*1e-3, Q=Q); Pf = lambda t: Ps * min(t / 0.005, 1)
        t, y, U, _ = simulate(p, Pf, 1.0, x0*1e-3, None, seed=7); return onset_growth(t, U, Pf)['success']
    if not on(hi): return dict(stage=stage, x0_mm=x0, Q=Q, Ps_th=np.nan)
    while hi - lo > 5.0:
        mid = 0.5 * (lo + hi)
        if on(mid): hi = mid
        else: lo = mid
    return dict(stage=stage, x0_mm=x0, Q=Q, Ps_th=0.5 * (lo + hi))
def h4(c):
    stage, m, sd = c; p = params(stage=stage); Pf = lambda t: m * PTH[stage] * min(t / 0.005, 1)
    t, y, U, _ = simulate(p, Pf, 2.0, XT, None, seed=sd); o = onset_growth(t, U, Pf); return dict(stage=stage, mult=m, seed=sd, **o)
if __name__ == '__main__':
    t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
    REL = (0.005, 0.020, 0.100); conds = [('easy', 0.0)] + [('hard', r) for r in REL]
    jobs = {}
    with Pool(8) as pool:
        jobs['nominal_sweep'] = pool.map_async(sweep_nominal, [(st, s, r, P) for st in 'BA' for s, r in conds for P in np.arange(round(0.75*PTH[st]), round(1.6*PTH[st]) + 1, 5)], chunksize=1)
        jobs['thresholds'] = pool.map_async(thr, [(st, s, r, i) for st in 'BA' for s, r in conds for i in range(20)], chunksize=1)
        jobs['delays'] = pool.map_async(delay, [(st, s, r, m, i) for st in 'BA' for s, r in conds for m in (1.05, 1.1, 1.2, 1.5) for i in range(50)], chunksize=2)
        jobs['h4'] = pool.map_async(h4, [(st, m, sd) for st in 'BA' for m in (1.03, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0) for sd in range(1, 11)], chunksize=1)
        P = [(st, s, Tr, r, 1.2, i) for st, N in (('B', 200), ('A', 100)) for Tr in (0.025, 0.2, 1.0) for s, r in conds for i in range(N)]
        P += [('B', s, 0.2, r, m, i) for m in (1.1, 1.5) for s, r in conds for i in range(100)]
        jobs['paths'] = pool.map_async(path, P, chunksize=2)
        X0S = [-0.05,0,0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]; QS = [0.5,0.65,0.8,1.0,1.2,1.4,1.7,2.0]
        jobs['map'] = pool.map_async(cell, [(st, x, q) for st in 'BA' for x in X0S for q in QS], chunksize=1)
        for k, j in jobs.items(): pd.DataFrame(j.get()).to_csv(f'{k}.csv', index=False); log(k, 'done')
