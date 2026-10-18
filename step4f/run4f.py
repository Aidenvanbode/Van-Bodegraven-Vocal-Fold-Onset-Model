import time, json, os, sys, numpy as np
from multiprocessing import Pool
import model_f as mf
XT = 0.2e-3; PTH = {'A': 259.8, 'B': 195.3125}; T_HOLD = 0.10; BASEHOLD = (2000.0, -0.05e-3)
def draw(i, kind='e'):
    if i < 0: return np.ones(7)
    if kind == 'e': return np.random.default_rng(1000 + i).uniform(0.9, 1.1, 7)
    return np.random.default_rng(9000 + i).uniform(0.7, 1.3, 8)   # wide: k1,k2,kc,m1,m2, zeta_open, zeta_contact, + x0 (0.9-1.1 below)
def setup(stage, i, kind='e', **extra):
    f = draw(i, kind); kw = {k: mf.BASE[k] * fi for k, fi in zip(['k1', 'k2', 'kc', 'm1', 'm2'], f[:5])}
    if kind == 'w':
        kw['zo'] = tuple(z * f[5] for z in mf.BASE['zo']); kw['zc'] = tuple(z * f[6] for z in mf.BASE['zc'])
        x0 = XT * (0.9 + 0.2 * (f[7] - 0.7) / 0.6); fP = 1.0
    else: x0 = XT * f[5]; fP = f[6]
    kw.update(extra); return mf.params(stage=stage, **kw), x0, fP
def const_run(stage, start, Ps, i, Trel, seed, hold=BASEHOLD, kind='e', Tafter=1.0, p=None, x0=None, keep=0.2):
    mf.set_hold(*hold)
    if p is None: p, x0, _ = setup(stage, i, kind)
    Pf = lambda t: Ps * min(t / 0.005, 1)
    if start == 'hard': w = mf.w_release(T_HOLD, Trel); T = T_HOLD + Trel + Tafter; t0 = T_HOLD
    else: w = None; T = Tafter + 0.1; t0 = 0.005
    t, y, U, _ = mf.simulate(p, Pf, T, x0, w, seed=seed); o = mf.detect_all(t, y, U, x0, Pf, keep=keep)
    for s in mf.RSTARS:
        k = f'R{s:g}'; o[f'delay_ms_{k}'] = (o[f't_on_{k}'] - t0) * 1e3
        if start == 'hard': o[f'delay_from_release_end_ms_{k}'] = (o[f't_on_{k}'] - T_HOLD - Trel) * 1e3
    if start == 'hard': o['max_gap_while_held_mm'] = float((x0 + np.minimum(y[0], y[2]))[t < T_HOLD].max() * 1e3)
    return o
def bisect(ok, lo, hi, tol):
    if not ok(hi): return np.nan
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(mid): hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)
# ---------------- job functions ----------------
def J_thr(stage, start, Trel, i, hold=list(BASEHOLD), rstar=0.1):
    key = f'succ_R{rstar:g}'
    th = bisect(lambda P: const_run(stage, start, P, i, Trel, 7 + i, tuple(hold))[key], 0.5 * PTH[stage], 1.6 * PTH[stage], 2.0)
    return dict(Ps_th=th)
def J_thr_v1(stage, start, Trel, i):
    """threshold with the OLD (4e) detector, re-computed in the same code for a like-for-like check."""
    th = bisect(lambda P: const_run(stage, start, P, i, Trel, 7 + i)['succ_v1_R0.1'], 0.5 * PTH[stage], 1.6 * PTH[stage], 2.0)
    return dict(Ps_th_v1=th)
def J_delay(stage, start, Trel, m, i, hold=list(BASEHOLD)):
    return const_run(stage, start, m * PTH[stage], i, Trel, 3000 + i, tuple(hold))
def J_path(stage, start, Tr, Trel, m, i, hold=list(BASEHOLD)):
    mf.set_hold(*hold); p, x0, fP = setup(stage, i); Pt = m * PTH[stage] * fP
    Pf = lambda t: Pt * min(t / Tr, 1); w = mf.w_release(Tr, Trel) if start == 'hard' else None
    T = Tr + Trel + 1.0; t, y, U, _ = mf.simulate(p, Pf, T, x0, w, seed=5000 + i); o = mf.detect_all(t, y, U, x0, Pf)
    o['max_gap_while_held_mm'] = float((x0 + np.minimum(y[0], y[2]))[t < Tr].max() * 1e3) if start == 'hard' else np.nan
    o['Ps_target'] = Pt
    for s in mf.RSTARS: o[f't_after_ramp_ms_R{s:g}'] = (o[f't_on_R{s:g}'] - Tr) * 1e3
    return o
def J_h4(stage, m, sd):
    p = mf.params(stage=stage); Pf = lambda t: m * PTH[stage] * min(t / 0.005, 1)
    t, y, U, _ = mf.simulate(p, Pf, 2.0, XT, None, seed=sd); return mf.detect_all(t, y, U, XT, Pf)
def J_cell(stage, x0, Q, crit):
    key = 'succ_R0.1' if crit == 'R' else 'succ_X2e-05'
    def on(Ps):
        p = mf.params(stage=stage, x0=x0 * 1e-3, Q=Q); Pf = lambda t: Ps * min(t / 0.005, 1)
        t, y, U, _ = mf.simulate(p, Pf, 1.0, x0 * 1e-3, None, seed=7); return mf.detect_all(t, y, U, x0 * 1e-3, Pf)[key]
    return dict(Ps_th=bisect(on, 0.0, 2000.0, 5.0))
def J_nonmono(stage, Trel, Ps, seed, Tafter):
    o = const_run(stage, 'hard', Ps, -1, Trel, seed, Tafter=Tafter)
    return {k: v for k, v in o.items() if not k.startswith(('t_on_X', 'Ps_on'))}
def onset_offset(p, x0, keep, tol, Pmax=3000.0):
    """onset: lowest constant Ps (5 ms rise, from rest, noise seed 7) giving sustained oscillation (fixed detector, R*=0.1).
    offset: lowest Ps at which an oscillation ESTABLISHED at 1.3 x onset survives after a 20 ms smooth step down."""
    Tafter = 1.0 + keep
    def on(P):
        Pf = lambda t: P * min(t / 0.005, 1); t, y, U, _ = mf.simulate(p, Pf, Tafter, x0, None, seed=7)
        return mf.detect_all(t, y, U, x0, Pf, keep=keep)['succ_R0.1']
    Pon = bisect(on, 0.0, Pmax, tol)
    if not np.isfinite(Pon): return dict(P_on=np.nan, P_off=np.nan, high_ok=False)
    Ph = 1.3 * Pon; T1 = 0.6
    def off(P):
        def Pf(t):
            if t < 0.005: return Ph * t / 0.005
            if t < T1: return Ph
            if t < T1 + 0.02: return P + (Ph - P) * 0.5 * (1 + np.cos(np.pi * (t - T1) / 0.02))
            return P
        t, y, U, _ = mf.simulate(p, Pf, T1 + Tafter, x0, None, seed=7)
        return mf.detect_all(t, y, U, x0, Pf, keep=keep)['succ_R0.1']
    high_ok = off(Ph)
    Poff = bisect(off, 0.0, Ph, tol) if high_ok else np.nan   # upper bracket = 1.3 x onset (known to sustain)
    return dict(P_on=Pon, P_off=Poff, high_ok=bool(high_ok))
def J_V3(stage, beta):
    p = mf.params(stage=stage, **mf.beta_kw(beta)); G, Plin = mf.linear_threshold_p(p, XT)
    o = onset_offset(p, XT, 0.2, 1.0); o.update(Gamma_th=G, P_lin=Plin, eq5=beta * Plin / (XT * 1.0)); return o
def J_V7(stage, Q):
    p = mf.params(stage=stage, Q=Q); G, Plin = mf.linear_threshold_p(p, XT); keep = max(0.2, 8 / (130 * Q))
    o = onset_offset(p, XT, keep, 0.5); o['P_lin'] = Plin; return o
def J_normal(stage, Q, Ps):
    p = mf.params(stage=stage, Q=Q); Pf = lambda t: Ps * min(t / 0.005, 1)
    t, y, U, _ = mf.simulate(p, Pf, 1.1, XT, None, seed=7); o = mf.detect_all(t, y, U, XT, Pf)
    return {k: o[k] for k in ('succ_R0.1', 'endR', 'F0')}
def J_robust(i):
    """wide parameter draw: own easy threshold (bisection, seed 7+i) then the race at 1.05/1.2/1.5 x that threshold."""
    p, x0, _ = setup('B', i, 'w')
    th = bisect(lambda P: const_run('B', 'easy', P, i, 0, 7 + i, p=p, x0=x0)['succ_R0.1'], 50.0, 1500.0, 2.0)
    out = dict(Ps_th=th, F0=np.nan)
    if not np.isfinite(th): return out
    for m in (1.05, 1.2, 1.5):
        for start, rel in (('easy', 0.0), ('hard', 0.005), ('hard', 0.020), ('hard', 0.100)):
            o = const_run('B', start, m * th, i, rel, 3000 + i, p=p, x0=x0); c = f'{m:g}_{start}{int(rel*1e3)}'
            out[f'succ_{c}'] = o['succ_R0.1']; out[f'delay_{c}'] = o['delay_ms_R0.1']
            if start == 'hard': out[f'gap_{c}'] = o['max_gap_while_held_mm']
            if start == 'easy' and m == 1.2: out['F0'] = o.get('F0', np.nan)
    return out
FUN = {k[2:]: v for k, v in globals().items() if k.startswith('J_')}
def jobs():
    import pandas as pd
    L = []; REL = (0.005, 0.020, 0.100); conds = [('easy', 0.0)] + [('hard', r) for r in REL]
    L += [('robust', [i]) for i in range(100)]
    L += [('V3', [st, b]) for st in 'BA' for b in (1.0, 0.9, 0.8, 0.72, 0.64)]
    L += [('V7', [st, q]) for st in 'BA' for q in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.71, 1.8)]
    L += [('normal', [st, 1.0, 800.0]) for st in 'BA']
    SET = [(500.0, -0.05e-3), (5000.0, -0.05e-3), (2000.0, -0.02e-3), (2000.0, -0.1e-3)]
    L += [('cell', [st, x, q, 'X']) for st in 'BA' for x in (0, 0.05, 0.2) for q in (0.5, 0.65, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0)]
    L += [('nonmono', ['A', r, float(P), sd, Ta]) for r in REL for P in range(250, 291, 2) for sd, Ta in ((7, 1.0), (7, 2.0), (8, 1.0))]
    # detector fix: re-run every 4e run that failed while its end level was >= 0.10 (exact same configuration and seed)
    E = '../step4e/'; d = pd.read_csv(E + 'delays.csv'); d = d[(~d.success) & (d.R_end >= 0.1)]
    L += [('delay', [r.stage, r.start, round(r.Trel_ms / 1e3, 3), float(r.mult), int(r.draw)]) for r in d.itertuples()]
    p = pd.read_csv(E + 'paths.csv'); p = p[(~p.success) & (p.R_end >= 0.1)]
    L += [('path', [r.stage, r.start, round(r.Tr_ms / 1e3, 3), round(r.Trel_ms / 1e3, 3), float(r.mult), int(r.draw)]) for r in p.itertuples()]
    h = pd.read_csv(E + 'h4.csv'); h = h[(~h.success) & (h.R_end >= 0.1)]
    L += [('h4', [r.stage, float(r.mult), int(r.seed)]) for r in h.itertuples()]
    # baseline race / H3 (Stage B, 20 / 40 draws) and the same for 2 extra values per hold axis
    L += [('delay', ['B', s, r, m, i]) for s, r in conds for m in (1.05, 1.2, 1.5) for i in range(20)]
    L += [('path', ['B', s, 0.2, r, 1.2, i]) for s, r in conds for i in range(40)]
    L += [('delay', ['B', 'hard', r, m, i, list(h)]) for h in SET for r in REL for m in (1.05, 1.2, 1.5) for i in range(20)]
    L += [('path', ['B', 'hard', 0.2, r, 1.2, i, list(h)]) for h in SET for r in REL for i in range(40)]
    # Jobs completed in the first (full-scale) pass before the scale was reduced: 620 thresholds, 160 old-detector
    # thresholds, 49 extra Stage B zero-gap cells. Their exact specs are stored so a fresh run reproduces every result.
    L += [tuple(j) for j in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'first_pass_jobs.json')))]
    seen = set(); out = []
    for j in L:
        k = key(j)
        if k not in seen: seen.add(k); out.append(j)
    return out
def key(j): return j[0] + '|' + json.dumps(j[1])
def work(j):
    t0 = time.time()
    try: r = FUN[j[0]](*j[1]); err = None
    except Exception as e: r = {}; err = repr(e)
    return dict(job=j[0], args=j[1], key=key(j), err=err, sec=time.time() - t0, **{k: (None if isinstance(v, float) and not np.isfinite(v) else (bool(v) if isinstance(v, (bool, np.bool_)) else (float(v) if isinstance(v, (np.floating, np.integer)) else v))) for k, v in r.items()})
if __name__ == '__main__':
    # resumable: jobs already in results.jsonl are skipped. Delete results.jsonl to recompute everything.
    out = 'results.jsonl'; done = set()
    if os.path.exists(out):
        for l in open(out): done.add(json.loads(l)['key'])
    L = [j for j in jobs() if key(j) not in done]; t0 = time.time(); n = len(L)
    print(f'{n} jobs to run ({len(done)} already done)', flush=True)
    with Pool(8) as pool, open(out, 'a') as f:
        for c, r in enumerate(pool.imap_unordered(work, L, chunksize=1), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if r['err']: print('ERR', r['key'], r['err'], flush=True)
            if c % 100 == 0 or c == n: print(f'[{time.time()-t0:.0f}s] {c}/{n}', flush=True)
    print('ALL DONE', flush=True)
