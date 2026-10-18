"""Step 4c model: Stage A (step4/model.py force table) or Stage B (Assaneo & Trevisan 2013, eqs. 6-12 aero forces),
same IF72 mechanics; options hc (contact-stiffness multiplier, h_i = hc*3*k_i) and EPS (smoothing width).
Noise as in Step 4b: extra pressure sigma*Ps*n(t) on mass 1 (force * lg*d1), n ~ N(0,1) at 2 kHz, interpolated."""
import sys; sys.path.insert(0, '../step4')
import numpy as np
from scipy.integrate import solve_ivp
from model import BASE, FS, params as _params, H, window_rms, classify, f0_of
MU = 1.8e-5; KS = 1.2; SIGMA = 0.005
def params(stage='A', hc=1.0, **kw):
    p = _params(**kw); p['stage'] = stage; p['hc'] = hc; return p
def noise_fun(seed, T, sigma=SIGMA):
    n = np.random.default_rng(seed).standard_normal(int(T * 2000) + 3); tt = np.arange(len(n)) / 2000.0
    return lambda t: sigma * np.interp(t, tt, n)
def aero_B(Ps, a1, a2, p):
    """Stage B. u_g from Bernoulli to the separation section plus Poiseuille drops (Lucero & Koenig 2005 style balance;
    A&T 2013 do not print the u_g equation). Pin uses Bernoulli Ps - rho u^2/(2 a1^2) (A&T print '+'; see report)."""
    lg, d1, d2, rho = p['lg'], p['d1'], p['d2'], p['rho']
    amn = 2 * lg * p['EPS']; a1 = np.sqrt(a1 * a1 + amn * amn); a2 = np.sqrt(a2 * a2 + amn * amn)  # regularize: areas never < 2*lg*EPS (avoids 1/a^3 blow-up at closure)
    sep1 = H(a2 - KS * a1, 2 * lg * p['EPS'])            # smooth switch: 1 -> separation at m1 exit
    ae = sep1 * a1 + (1 - sep1) * a2
    R = 12 * MU * lg**2 * (d1 / a1**3 + (1 - sep1) * d2 / a2**3)   # viscous resistance
    A = rho / (2 * ae**2)                                 # A u^2 + R u - Ps = 0
    u = (-R + np.sqrt(R * R + 4 * A * max(Ps, 0))) / (2 * A)
    Pin = Ps - rho * u * u / (2 * a1**2); P12 = Pin - 12 * MU * u * d1 * lg**2 / a1**3
    P21 = (1 - sep1) * 12 * MU * u * d2 * lg**2 / a2**3
    return lg * d1 * (Pin + P12) / 2, lg * d2 * P21 / 2, u
def make_rhs(p, Ps_fun, x0_fun, nf=None):
    m1, m2, k1, k2, kc = p['m1e'], p['m2e'], p['k1e'], p['k2e'], p['kce']
    lg, d1, d2, eps, ek, eh, hc = p['lg'], p['d1'], p['d2'], p['EPS'], p['eta_k'], p['eta_h'], p['hc']
    sq1, sq2 = 2 * np.sqrt(m1 * k1), 2 * np.sqrt(m2 * k2); B = p['stage'] == 'B'
    def rhs(t, y):
        x1, v1, x2, v2 = y; Ps = Ps_fun(t); x0 = x0_fun(t)
        a1 = 2 * lg * (x0 + x1); a2 = 2 * lg * (x0 + x2)
        c1 = H(-(x0 + x1), eps); c2 = H(-(x0 + x2), eps); o1 = 1 - c1; o2 = 1 - c2
        a1p = max(a1, 1e-12); a2p = max(a2, 1e-12)
        if B:
            fa1, fa2, _ = aero_B(Ps, a1p, a2p, p)
            f1 = o1 * o2 * fa1 + c1 * Ps * lg * d1 + o1 * c2 * Ps * lg * d1; f2 = o1 * o2 * fa2 + o1 * c2 * Ps * lg * d2
        else:
            amin = min(a1p, a2p)
            f1 = o1 * o2 * Ps * lg * d1 * (1 - min(amin / a1p, 2.0) ** 2) + c1 * Ps * lg * d1 + o1 * c2 * Ps * lg * d1
            f2 = o1 * c2 * Ps * lg * d2
        if nf is not None: f1 += Ps * nf(t) * lg * d1
        s1 = k1 * x1 * (1 + ek * x1 * x1) + c1 * hc * 3 * k1 * (x1 + x0) * (1 + eh * (x1 + x0) ** 2)
        s2 = k2 * x2 * (1 + ek * x2 * x2) + c2 * hc * 3 * k2 * (x2 + x0) * (1 + eh * (x2 + x0) ** 2)
        r1 = (o1 * p['zo'][0] + c1 * p['zc'][0]) * sq1; r2 = (o2 * p['zo'][1] + c2 * p['zc'][1]) * sq2
        return [v1, (f1 - s1 - r1 * v1 - kc * (x1 - x2)) / m1, v2, (f2 - s2 - r2 * v2 - kc * (x2 - x1)) / m2]
    return rhs
def simulate(p, Ps_fun, T, x0_fun=None, y0=None, seed=7, sigma=SIGMA):
    if x0_fun is None: x0c = p['x0']; x0_fun = lambda t: x0c
    nf = noise_fun(seed, T, sigma) if sigma > 0 else None
    if y0 is None: y0 = [0, 0, 0, 0] if sigma > 0 else [1e-6, 0, 1e-6, 0]
    te = np.arange(0, T, 1 / FS)
    sol = solve_ivp(make_rhs(p, Ps_fun, x0_fun, nf), (0, T), y0, method='LSODA', rtol=1e-8, atol=1e-11, max_step=2e-5, t_eval=te)
    x0v = np.array([x0_fun(t) for t in sol.t]); Psv = np.array([Ps_fun(t) for t in sol.t])
    a1 = 2 * p['lg'] * (x0v + sol.y[0]); a2 = 2 * p['lg'] * (x0v + sol.y[2])
    if p['stage'] == 'B':
        U = np.array([aero_B(P, A1, A2, p)[2] if min(A1, A2) > 0 else 0.0 for P, A1, A2 in zip(Psv, np.clip(a1, 1e-12, None), np.clip(a2, 1e-12, None))])
    else:
        U = np.clip(np.minimum(a1, a2), 0, None) * np.sqrt(2 * np.clip(Psv, 0, None) / p['rho'])
    return sol.t, sol.y, U, Psv
def onset_const(Ps, T=1.0, seed=7, sigma=SIGMA, **kw):
    p = params(**kw); t, y, U, _ = simulate(p, lambda tt: Ps * min(1.0, tt / 0.005), T, seed=seed, sigma=sigma)
    on, rms = classify(t, U); return on, rms, (f0_of(t, U) if on else np.nan)
def bisect(lo=0.0, hi=2000.0, tol=5.0, **kw):
    if not onset_const(hi, **kw)[0]: return np.nan
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if onset_const(mid, **kw)[0]: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)
def onset_time(t, U, Pf):
    tc, r = window_rms(U); above = r > 1e-6
    if not above[-1]: return np.nan, np.nan
    k = len(above) - 1
    while k > 0 and above[k - 1]: k -= 1
    return tc[k] - 0.005, Pf(tc[k] - 0.005)
