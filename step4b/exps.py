import time, pandas as pd
from multiprocessing import Pool
from common import *
PTH = 259.8; XT = 0.2e-3; XP = -0.05e-3
X0S = [-0.05,0,0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]; QS = [0.5,0.65,0.8,1.0,1.2,1.4,1.7,2.0]
def cell(c):
    x0, Q = c; P, n = bisect_n(0, 2000, 10.0, x0=x0*1e-3, Q=Q); return dict(x0_mm=x0, Q=Q, Ps_th=P, runs=n)
def path(start, Tr, Pt, xt):
    """Linear pressure ramp 0->Pt over Tr for BOTH starts (same shape and duration -> no ramp confound).
    hard: pressed x0=-0.05 mm until ramp end, then 1 ms release to xt (Step 3 H1). easy: fixed slight gap xt (Step 3 E2 gap)."""
    Pf = lambda t: Pt * min(t / Tr, 1.0)
    if start == 'hard': Xf = lambda t: XP + (xt - XP) * min(max((t - Tr) / 1e-3, 0), 1)
    else: Xf = lambda t: xt
    return Pf, Xf
def run(c):
    start, Tr, mult, i = c; rng = np.random.default_rng(1000 + i); f = rng.uniform(0.9, 1.1, 7)
    kw = {k: BASE[k] * fi for k, fi in zip(['k1','k2','kc','m1','m2'], f[:5])}; p = params(**kw)
    Pt = mult * PTH * f[6]; xt = XT * f[5]; Pf, Xf = path(start, Tr, Pt, xt); T = Tr + 1.0
    t, y, U, Ps = simulate(p, noisy(Pf, 5000 + i, T), T, x0_fun=Xf, y0=[0,0,0,0])
    on, rms = classify(t, U); ton, Pon = onset_time(t, U, Pf)
    # contact (effort proxy): max penetration depth of mass 1
    X = np.array([Xf(tt) for tt in t]); pen = np.clip(-(X + y[0]), 0, None).max()
    return dict(start=start, Tr_ms=Tr*1e3, mult=mult, draw=i, success=on, rms_cm3s=rms*1e6, t_onset_ms=ton*1e3,
                Ps_at_onset=Pon, Ps_target=Pt, max_pen_um=pen*1e6)
def h4(c):
    m, sd = c; t, y, U, _ = simulate(params(), noisy(lambda t: m*PTH*min(1, t/0.005), sd, 2.0), 2.0, y0=[0,0,0,0])
    ton, _ = onset_time(t, U, lambda t: 0); return dict(mult=m, seed=sd, t_onset_ms=ton*1e3, success=classify(t, U)[0])
if __name__ == '__main__':
    import sys; t0 = time.time(); N = int(sys.argv[1])
    with Pool(8) as pool:
        H = pd.DataFrame(pool.map(h4, [(m, s) for m in (1.03,1.05,1.1,1.2,1.3,1.5,2.0) for s in (1,2,3)])); H.to_csv('h4_const.csv', index=False)
        print('h4', time.time()-t0, flush=True)
        R = pd.DataFrame(pool.map(run, [(s, Tr, m, i) for m in (1.2, 1.1, 1.5) for s in ("hard","easy") for Tr in (0.025, 0.2) for i in range(N if m == 1.2 else N//2)], chunksize=4))
        R.to_csv('exp23_paths.csv', index=False); print('paths', time.time()-t0, flush=True)
        M = pd.DataFrame(pool.map(cell, [(x, q) for x in X0S for q in QS], chunksize=1)); M.to_csv('exp1_onset_map.csv', index=False)
    print('done', time.time()-t0)
