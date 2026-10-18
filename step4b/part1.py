# threshold (noise vs none), hysteresis (stepped, noise, NO nudge), slow-ramp test
import time, pandas as pd
from multiprocessing import Pool
from common import *
def thr(cfg):
    sig, seed = cfg; P, n = bisect_n(100, 500, 1.0, sigma=sig, seed=seed); return dict(sigma=sig, seed=seed, Ps_th=P)
def hyst(dP, hold=0.3, sig=SIGMA):
    p = params(); y = [0, 0, 0, 0]; rows = []
    lv = list(np.arange(200, 321, dP)); lv = lv + lv[::-1][1:]
    for k, Ps in enumerate(lv):
        Pf = noisy(lambda t, Ps=Ps: Ps, 100 + k, hold, sig)
        t, Y, U, _ = simulate(p, Pf, hold, y0=list(y)); on, rms = classify(t, U, t_keep=0.1)
        rows.append(dict(dP=dP, sigma=sig, dir='up' if k <= len(lv) // 2 else 'down', Ps=Ps, on=on, rms=rms)); y = Y[:, -1]
    return rows
def ramp(cfg):
    Tr, Q, sig = cfg; p = params(Q=Q); Pt = 800.0
    Pf0 = lambda t: Pt * 0.5 * (1 - np.cos(np.pi * min(t / Tr, 1))); T = Tr + 1.0
    t, y, U, _ = simulate(p, noisy(Pf0, 5, T, sig), T, y0=[0,0,0,0] if sig else None)
    on, rms = classify(t, U); ton, Pon = onset_time(t, U, Pf0)
    return dict(Tr_ms=Tr*1e3, Q=Q, sigma=sig, success=on, rms_cm3s=rms*1e6, t_onset_ms=ton*1e3, Ps_at_onset=Pon)
if __name__ == '__main__':
    t0 = time.time()
    with Pool(8) as pool:
        a = pool.map_async(thr, [(0, 0)] + [(s, sd) for s in (0.001, 0.005, 0.01) for sd in (1, 2)])
        h = pool.map_async(hyst, [10, 5])
        r = pool.map_async(ramp, [(Tr, Q, s) for Tr in (0.4, 1.0) for Q in (1.0, 1.4) for s in (0, SIGMA)])
        A = pd.DataFrame(a.get()); Hd = pd.DataFrame([x for l in h.get() for x in l]); R = pd.DataFrame(r.get())
    A.to_csv('p1_threshold.csv', index=False); Hd.to_csv('p1_hyst_steps.csv', index=False); R.to_csv('p1_slow_ramps.csv', index=False)
    for dP, g in Hd.groupby('dP'):
        up = g[g.dir == 'up']; dn = g[g.dir == 'down']; on = up[up.on].Ps.min(); off = dn[dn.on].Ps.min()
        print('hyst dP', dP, 'onset', on, 'lowest-on down', off, 'ratio', off / on)
    print(A.to_string()); print(R.to_string()); print(time.time() - t0)
