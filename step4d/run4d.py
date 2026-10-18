"""Step 4d runs. Re-runs path experiments with full onset logging (2.5 ms windows), AC/DC cut-offs, release-duration,
no-ramp release (kick) and minimal-smoothing tests. Model = ../step4c/model_c.py unchanged."""
import sys; sys.path.insert(0, '../step4c')
import time, numpy as np, pandas as pd
from multiprocessing import Pool
from model_c import *
XT, XP = 0.2e-3, -0.05e-3; PTH = {'A': 259.8, 'B': 195.3125}; CUTS = (0.0, 0.01, 0.02, 0.05, 0.10)
WIN = 0.010  # 10 ms (>= 1 cycle); 2.5 ms windows split cycles and break the "final above-threshold stretch"
def onset_info(t, U, Pf, Xf, Tstart, rel):
    n = int(WIN * FS); out = {}
    k = np.arange(0, len(U) - n + 1, n); seg = np.array([U[i:i + n] for i in k]); tt = np.arange(n)
    rms = np.array([np.std(s - np.polyval(np.polyfit(tt, s, 1), tt)) for s in seg]); dc = seg.mean(1) + 1e-30
    m = t >= t[-1] - 0.2; u = U[m] - U[m].mean(); r_end = np.sqrt(np.mean(u**2)); h = len(u)//2
    sus = np.sqrt(np.mean(u[h:]**2)) / (np.sqrt(np.mean(u[:h]**2)) + 1e-30) > 0.9; acdc_end = r_end / (U[m].mean() + 1e-30)
    for c in CUTS:
        tag = f'{int(c*100)}'; above = (rms > 1e-6) & (rms / dc > c)
        out[f'succ_{tag}'] = bool(r_end > 1e-6 and sus and acdc_end > c)
        if above[-1]:
            j = len(above) - 1
            while j > 0 and above[j - 1]: j -= 1
            ton = k[j] / FS; out[f't_on_{tag}'] = ton * 1e3; out[f'Ps_on_{tag}'] = Pf(ton)
            if c == 0.02:
                out['gap_on_mm'] = Xf(ton) * 1e3; out['amp_on_cm3s'] = rms[j] * 1e6
                te = ton + WIN  # window spans [ton, te)
                out['phase'] = 'pressed' if te <= Tstart else ('release' if ton < Tstart + rel else 'after')
                out['t_on_mid_ms'] = (ton + WIN / 2) * 1e3; out['Ps_on_mid'] = Pf(ton + WIN / 2)
        else: out[f't_on_{tag}'] = np.nan; out[f'Ps_on_{tag}'] = np.nan
    out['acdc_end'] = acdc_end; out['rms_end_cm3s'] = r_end * 1e6
    return out
def path_run(c):
    exp, stage, start, Tr, mult, i, rel, eps = c
    rng = np.random.default_rng(1000 + i); f = rng.uniform(0.9, 1.1, 7)
    kw = {k: BASE[k] * fi for k, fi in zip(['k1','k2','kc','m1','m2'], f[:5])}; kw['EPS'] = eps
    p = params(stage=stage, **kw); Pt = mult * PTH[stage] * f[6]; xt = XT * f[5]
    Pf = lambda t: Pt * min(t / Tr, 1.0)
    Xf = (lambda t: XP + (xt - XP) * min(max((t - Tr) / rel, 0), 1)) if start == 'hard' else (lambda t: xt)
    t, y, U, _ = simulate(p, Pf, Tr + 1.0, x0_fun=Xf, seed=5000 + i)
    X = np.array([Xf(s) for s in t]); open_pressed = bool(((X + np.minimum(y[0], y[2]) > 0) & (t < Tr)).any())
    tb = t < Tr; first_open = t[tb][(X[tb] + np.minimum(y[0][tb], y[2][tb]) > 0)].min()*1e3 if open_pressed else np.nan
    d = dict(exp=exp, stage=stage, start=start, Tr_ms=Tr*1e3, mult=mult, draw=i, rel_ms=rel*1e3, eps_um=eps*1e6, Ps_target=Pt,
             blown_open_while_pressed=open_pressed, t_first_open_ms=first_open, Ps_first_open=Pf(first_open/1e3) if open_pressed else np.nan)
    d.update(onset_info(t, U, Pf, Xf, Tr if start == 'hard' else 0.0, rel)); return d
def kick_run(c):
    """No ramp: Ps held constant from t=0 (5 ms rise). hard: pressed until t=0.15 s then release over rel; easy: open 0.2 mm all along.
    Run 0.8 s after release. Tests whether the release starts oscillation below/at the static threshold."""
    stage, start, mult, i, rel = c
    rng = np.random.default_rng(1000 + i); f = rng.uniform(0.9, 1.1, 7)
    kw = {k: BASE[k] * fi for k, fi in zip(['k1','k2','kc','m1','m2'], f[:5])}; p = params(stage=stage, **kw)
    Pt = mult * PTH[stage]; xt = XT * f[5]; T0 = 0.15
    Pf = lambda t: Pt * min(t / 0.005, 1.0)
    Xf = (lambda t: XP + (xt - XP) * min(max((t - T0) / rel, 0), 1)) if start == 'hard' else (lambda t: xt)
    t, y, U, _ = simulate(p, Pf, T0 + 0.8, x0_fun=Xf, seed=7000 + i)
    d = dict(exp='kick', stage=stage, start=start, mult=mult, draw=i, rel_ms=rel*1e3)
    o = onset_info(t, U, Pf, Xf, T0, rel); d.update(o)
    for c in CUTS:  # time after release (hard) / after pressure on (easy, measured from T0 too for equal reference)
        d[f't_after_T0_{int(c*100)}'] = o[f't_on_{int(c*100)}'] - T0*1e3
    return d
if __name__ == '__main__':
    t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
    P = []
    for st in 'AB':
        for m in ((1.1, 1.2, 1.5) if st == 'A' else (1.2,)):
            N = 200 if m == 1.2 else 100
            P += [('main', st, s, Tr, m, i, 1e-3, 1e-6) for s in ('hard','easy') for Tr in (0.025, 0.2) for i in range(N)]
        P += [('release', st, 'hard', Tr, 1.2, i, rel, 1e-6) for rel in (0.005, 0.020) for Tr in (0.025, 0.2) for i in range(200)]
        P += [('eps0.05', st, s, Tr, 1.2, i, 1e-3, 0.05e-6) for s in ('hard','easy') for Tr in (0.025, 0.2) for i in range(200)]
    K = [(st, s, m, i, rel) for st in 'AB' for m in (0.8, 0.9, 1.0, 1.1, 1.2) for i in range(50) for s, rel in (('hard', 1e-3), ('hard', 0.020), ('easy', 1e-3))]
    log('paths', len(P), 'kick', len(K))
    with Pool(8) as pool:
        kr = pool.map_async(kick_run, K, chunksize=4); pr = pool.map_async(path_run, P, chunksize=4)
        pd.DataFrame(kr.get()).to_csv('kick_runs.csv', index=False); log('kick done')
        pd.DataFrame(pr.get()).to_csv('path_runs.csv', index=False); log('paths done')
