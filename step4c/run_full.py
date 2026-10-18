"""Step 4c full-completion runs (after the steering message: no cuts)."""
import time, pandas as pd, numpy as np
from multiprocessing import Pool
from model_c import *
from run_c import path_run, gap, SENS
def sweep(c):
    x0, Q, Ps, seed = c; p = params(x0=x0*1e-3, Q=Q)
    t, y, U, _ = simulate(p, lambda tt: Ps * min(1.0, tt / 0.005), 1.0, seed=seed); on, rms = classify(t, U)
    m = t >= t[-1] - 0.2
    return dict(x0_mm=x0, Q=Q, Ps=Ps, seed=seed, on=on, rms_cm3s=rms*1e6, Udc_cm3s=U[m].mean()*1e6, F0=f0_of(t, U) if on else np.nan)
if __name__ == '__main__':
    t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
    X = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]; QS = [0.5,0.65,0.8,1.0,1.2,1.4,1.7,2.0]
    PthB = 195.3125
    with Pool(8) as pool:
        G = pool.map_async(gap, [(s, x, sd) for s in 'B' for x in (-0.05,0,0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.6) for sd in (7, 8)], chunksize=1)
        BP = pool.map_async(path_run, [('B', {}, 'stageB', s, Tr, 1.2*PthB, i) for s in ('hard','easy') for Tr in (0.025, 0.2) for i in range(100, 200)], chunksize=2)
        SP = pool.map_async(path_run, [('A', kw, n, s, 0.025, 1.2*259.8, i) for n, kw in SENS for s in ('hard','easy') for i in range(40, 200)]
                                    + [('A', kw, n, s, 0.2, 1.2*259.8, i) for n, kw in SENS + [('baseline', {})] for s in ('hard','easy') for i in range(200)], chunksize=2)
        SW = pool.map_async(sweep, [(x, q, P, sd) for sd in (1,2,3) for x in X for q in QS for P in range(100, 2001, 50)], chunksize=4)
        pd.DataFrame(G.get()).to_csv('stageB_gap_threshold_tol5.csv', index=False); log('gap')
        pd.DataFrame(BP.get()).to_csv('stageB_paths_100_199.csv', index=False); log('Bpaths')
        pd.DataFrame(SP.get()).to_csv('sens_paths_extra.csv', index=False); log('sens')
        S = pd.DataFrame(SW.get()); S.to_csv('wide_sweep_full.csv', index=False); log('sweep')
        # 10 Pa refinement around lowest onset (AC/DC criterion), 3 seeds
        S['ok'] = S.on & (S.rms_cm3s / S.Udc_cm3s > 0.02)
        ref = []
        for (x, q), g in S.groupby(['x0_mm', 'Q']):
            lo = g[g.ok].Ps.min()
            if np.isfinite(lo): ref += [(x, q, P, sd) for sd in (1,2,3) for P in range(int(lo) - 50, int(lo) + 1, 10) if P >= 100]
        pd.DataFrame(pool.map(sweep, ref, chunksize=4)).to_csv('wide_sweep_refine10.csv', index=False); log('refine')
