"""Step 4 model: symmetric two-mass model (Stage A), derived from ../vf_starter.py.
Options: flow_mode='separation' (default, IF72/SH95 force table) or 'symmetric'
(diagnostic: f1 = Ps*lg*d1*(1-(a2/a1)^2) also when divergent -> no half-cycle switch).
x0 and Ps can be functions of time. Q scales k_i, kc by Q and m_i by 1/Q (IF72/Assaneo&Trevisan)."""
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(m1=0.125e-3, m2=0.025e-3, k1=80.0, k2=8.0, kc=25.0, eta_k=1e6, eta_h=5e6,
            zo=(0.1, 0.6), zc=(1.1, 1.9), d1=0.25e-2, d2=0.05e-2, lg=1.4e-2, x0=0.02e-2,
            rho=1.2, EPS=1e-6, Q=1.0, flow_mode='separation')
FS = 44100.0

def params(**kw):
    p = dict(BASE); p.update(kw)
    Q = p['Q']
    p['m1e'], p['m2e'] = p['m1'] / Q, p['m2'] / Q
    p['k1e'], p['k2e'], p['kce'] = p['k1'] * Q, p['k2'] * Q, p['kc'] * Q
    return p

def H(z, eps):
    return 0.5 * (1.0 + np.tanh(z / eps))

def make_rhs(p, Ps_fun, x0_fun):
    m1, m2, k1, k2, kc = p['m1e'], p['m2e'], p['k1e'], p['k2e'], p['kce']
    lg, d1, d2, eps, ek, eh = p['lg'], p['d1'], p['d2'], p['EPS'], p['eta_k'], p['eta_h']
    sq1, sq2 = 2 * np.sqrt(m1 * k1), 2 * np.sqrt(m2 * k2)
    sym = p['flow_mode'] == 'symmetric'
    def rhs(t, y):
        x1, v1, x2, v2 = y
        Ps = Ps_fun(t); x0 = x0_fun(t)
        a1 = 2 * lg * (x0 + x1); a2 = 2 * lg * (x0 + x2)
        c1 = H(-(x0 + x1), eps); c2 = H(-(x0 + x2), eps); o1 = 1 - c1; o2 = 1 - c2
        a1p = a1 if a1 > 1e-12 else 1e-12; a2p = a2 if a2 > 1e-12 else 1e-12
        amin = a2p if sym else min(a1p, a2p)
        f1 = o1 * o2 * Ps * lg * d1 * (1 - min(amin / a1p, 2.0) ** 2) + c1 * Ps * lg * d1 + o1 * c2 * Ps * lg * d1
        f2 = o1 * c2 * Ps * lg * d2
        s1 = k1 * x1 * (1 + ek * x1 * x1) + c1 * 3 * k1 * (x1 + x0) * (1 + eh * (x1 + x0) ** 2)
        s2 = k2 * x2 * (1 + ek * x2 * x2) + c2 * 3 * k2 * (x2 + x0) * (1 + eh * (x2 + x0) ** 2)
        r1 = (o1 * p['zo'][0] + c1 * p['zc'][0]) * sq1; r2 = (o2 * p['zo'][1] + c2 * p['zc'][1]) * sq2
        return [v1, (f1 - s1 - r1 * v1 - kc * (x1 - x2)) / m1, v2, (f2 - s2 - r2 * v2 - kc * (x2 - x1)) / m2]
    return rhs

def simulate(p, Ps_fun, T, x0_fun=None, y0=None, rtol=1e-8, atol=1e-11, max_step=2e-5, method='LSODA'):
    if x0_fun is None:
        x0c = p['x0']; x0_fun = lambda t: x0c
    if y0 is None: y0 = [1e-6, 0, 1e-6, 0]
    t_eval = np.arange(0, T, 1 / FS)
    sol = solve_ivp(make_rhs(p, Ps_fun, x0_fun), (0, T), y0, method=method, rtol=rtol, atol=atol,
                    max_step=max_step, t_eval=t_eval)
    x0v = np.array([x0_fun(t) for t in sol.t]); Psv = np.array([Ps_fun(t) for t in sol.t])
    a = 2 * p['lg'] * (x0v + np.minimum(sol.y[0], sol.y[2]))
    U = np.clip(a, 0, None) * np.sqrt(2 * np.clip(Psv, 0, None) / p['rho'])
    return sol.t, sol.y, U, Psv

def window_rms(U, fs=FS, win=1 / 100):
    n = int(win * fs); out = []; tc = []
    for i in range(0, len(U) - n + 1, n):
        seg = U[i:i + n]; tt = np.arange(n); c = np.polyfit(tt, seg, 1)
        out.append(np.std(seg - np.polyval(c, tt))); tc.append((i + n / 2) / fs)
    return np.array(tc), np.array(out)

def classify(t, U, t_keep=0.2, thr=1e-6):
    m = t >= t[-1] - t_keep; u = U[m] - U[m].mean(); rms = np.sqrt(np.mean(u ** 2))
    h = len(u) // 2; r1 = np.sqrt(np.mean(u[:h] ** 2)) + 1e-30; r2 = np.sqrt(np.mean(u[h:] ** 2))
    return bool(rms > thr and r2 / r1 > 0.9), rms

def f0_of(t, U, t_keep=0.2):
    m = t >= t[-1] - t_keep; u = U[m] - U[m].mean()
    zc = np.where((u[:-1] < 0) & (u[1:] >= 0))[0]
    return FS / np.mean(np.diff(zc)) if len(zc) > 2 else np.nan

def onset_const(Ps, T=1.0, **kw):
    sim_kw = {k: kw.pop(k) for k in list(kw) if k in ('rtol', 'atol', 'max_step', 'y0', 'method')}
    thr = kw.pop('thr', 1e-6)
    p = params(**kw)
    t, y, U, _ = simulate(p, lambda tt: Ps * min(1.0, tt / 0.005), T, **sim_kw)
    on, rms = classify(t, U, thr=thr)
    return on, rms, (f0_of(t, U) if on else np.nan)

def threshold_bisect(lo=0.0, hi=2000.0, tol=5.0, T=1.0, **kw):
    """Assumes monotone onset in Ps. Returns (Ps_th or nan if no onset at hi, n_runs)."""
    n = 1
    if not onset_const(hi, T=T, **dict(kw))[0]: return np.nan, n
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); n += 1
        if onset_const(mid, T=T, **dict(kw))[0]: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi), n

def linear_threshold(Q=1.0, x0=0.02e-2):
    from scipy.optimize import brentq
    p = params(Q=Q, x0=x0); m1, m2, k1, k2, kc = p['m1e'], p['m2e'], p['k1e'], p['k2e'], p['kce']
    r1 = 2 * p['zo'][0] * np.sqrt(m1 * k1); r2 = 2 * p['zo'][1] * np.sqrt(m2 * k2)
    def rh(G):
        mu1, mu2 = r1 / m1, r2 / m2; w1, w2 = (k1 + kc - G) / m1, (k2 + kc) / m2
        kap = kc * (kc - G) / (m1 * m2)
        a1, a2, a3, a4 = mu1 + mu2, w1 + w2 + mu1 * mu2, mu1 * w2 + mu2 * w1, w1 * w2 - kap
        return a1 * a2 * a3 - a3 ** 2 - a1 ** 2 * a4
    G = np.linspace(0.1, 500, 5000); v = np.array([rh(g) for g in G])
    i = np.where(np.sign(v[1:]) != np.sign(v[:-1]))[0][0]
    Gth = brentq(rh, G[i], G[i + 1])
    return Gth, Gth * x0 / (2 * p['d1'] * p['lg'])
