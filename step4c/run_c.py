import time, pandas as pd, numpy as np
from multiprocessing import Pool
from model_c import *
XT, XP = 0.2e-3, -0.05e-3
def sweep(c):
    x0, Q, Ps, seed = c; on, rms, f0 = onset_const(Ps, T=0.8, seed=seed, x0=x0*1e-3, Q=Q)
    return dict(x0_mm=x0, Q=Q, Ps=Ps, seed=seed, on=on, rms_cm3s=rms*1e6, F0=f0)
def gap(c):
    stage, x0, seed = c; return dict(stage=stage, x0_mm=x0, seed=seed, Ps_th=bisect(0, 2000, 10.0, stage=stage, x0=x0*1e-3, seed=seed))
def sens_thr(c):
    name, kw = c; return dict(setting=name, Ps_th_pressed=bisect(0, 1000, 5.0, x0=XP, **kw))
def path_run(c):
    stage, extra, name, start, Tr, Pmean, i = c
    rng = np.random.default_rng(1000 + i); f = rng.uniform(0.9, 1.1, 7)
    kw = {k: BASE[k] * fi for k, fi in zip(['k1','k2','kc','m1','m2'], f[:5])}; kw.update(extra); p = params(stage=stage, **kw)
    Pt = Pmean * f[6]; xt = XT * f[5]; Pf = lambda t: Pt * min(t / Tr, 1.0)
    Xf = (lambda t: XP + (xt - XP) * min(max((t - Tr) / 1e-3, 0), 1)) if start == 'hard' else (lambda t: xt)
    t, y, U, _ = simulate(p, Pf, Tr + 1.0, x0_fun=Xf, seed=5000 + i)
    on, rms = classify(t, U); ton, Pon = onset_time(t, U, Pf)
    return dict(stage=stage, setting=name, start=start, Tr_ms=Tr*1e3, draw=i, success=on, Ps_at_onset=Pon, t_onset_ms=ton*1e3, Ps_target=Pt)
def hyst(c):
    stage, Pth, dP = c; p = params(stage=stage); y = [0,0,0,0]; rows = []
    lv = list(np.arange(round(Pth/5)*5 - 50, round(Pth/5)*5 + 51, dP)); lv = lv + lv[::-1][1:]
    for k, Ps in enumerate(lv):
        t, Y, U, _ = simulate(p, lambda t, Ps=Ps: Ps, 0.3, y0=list(y), seed=100 + k); on, rms = classify(t, U, t_keep=0.1)
        rows.append(dict(stage=stage, dP=dP, dir='up' if k <= len(lv)//2 else 'down', Ps=Ps, on=on, rms=rms)); y = Y[:, -1]
    return rows
SENS = [('eps0.25um', dict(EPS=0.25e-6)), ('eps4um', dict(EPS=4e-6)), ('hc0.5', dict(hc=0.5)), ('hc2', dict(hc=2.0))]
if __name__ == '__main__':
    t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
    X = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]; QS = [0.5,0.65,0.8,1.0,1.2,1.4,1.7,2.0]; PS = list(range(100, 2001, 50))
    with Pool(8) as pool:
        G = pool.map_async(gap, [(s, x, 7) for s in 'B' for x in (-0.05,0,0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.6)] + [('B', 0.2, 8), ('A', 0.2, 8)], chunksize=1)
        ST = pool.map_async(sens_thr, SENS, chunksize=1)
        SP = pool.map_async(path_run, [('A', kw, n, s, 0.025, 1.2*259.8, i) for n, kw in SENS for s in ('hard','easy') for i in range(40)], chunksize=2)
        SW = pool.map_async(sweep, [(x, q, P, 1) for x in X for q in QS for P in PS] + [(x, q, P, 2) for x in X for q in (1.0, 1.7) for P in PS[::2]], chunksize=4)
        G = pd.DataFrame(G.get()); G.to_csv('stageB_gap_threshold.csv', index=False); log(G.to_string())
        PthB = G[(G.stage=='B') & (G.x0_mm==0.2)].Ps_th.mean(); log('PthB', PthB)
        HY = pool.map_async(hyst, [('B', PthB, 5)])
        BP = pool.map_async(path_run, [('B', {}, 'stageB', s, Tr, 1.2*PthB, i) for s in ('hard','easy') for Tr in (0.025, 0.2) for i in range(100)], chunksize=2)
        pd.DataFrame(ST.get()).to_csv('sens_pressed_threshold.csv', index=False); pd.DataFrame(SP.get()).to_csv('sens_paths.csv', index=False); log('sens done')
        pd.DataFrame([r for l in HY.get() for r in l]).to_csv('stageB_hyst.csv', index=False); log('hyst done')
        pd.DataFrame(SW.get()).to_csv('wide_sweep.csv', index=False); log('sweep done')
        pd.DataFrame(BP.get()).to_csv('stageB_paths.csv', index=False); log('B paths done')
