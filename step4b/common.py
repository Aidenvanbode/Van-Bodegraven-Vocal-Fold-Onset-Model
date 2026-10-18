"""Step 4b: Stage A model from ../step4/model.py + subglottal pressure noise.
Ps_noisy(t) = Ps(t) * (1 + SIGMA * n(t)); n = unit-variance Gaussian sampled at 2 kHz (fixed seed),
linearly interpolated (keeps the RHS continuous for LSODA). SIGMA = 0.5 % (inside the 0.1-1 % brief)."""
import sys; sys.path.insert(0, '../step4')
import numpy as np
from model import *
SIGMA = 0.005; NFS = 2000.0
def noise_fun(seed, T, sigma=SIGMA):
    n = np.random.default_rng(seed).standard_normal(int(T * NFS) + 3); tt = np.arange(len(n)) / NFS
    return lambda t: 1.0 + sigma * np.interp(t, tt, n)
import model as _m
_NOISE = {}
def noisy(Pf, seed, T, sigma=SIGMA):
    """Returns Pf unchanged, but registers a noise process used by the patched RHS below.
    Reason: in this model the flow force on mass 1 is Ps*lg*d1*(1-(amin/a1)^2), which is exactly 0 at rest
    (a1=a2), so noise multiplying Ps cannot excite the folds. The fluctuation dP=sigma*Ps*n(t) is therefore
    applied as an extra pressure on the lower-mass surface (force dP*lg*d1, plus dP*lg*d2 on mass 2 is omitted)."""
    _NOISE['f'] = noise_fun(seed, T, sigma) if sigma > 0 else None; return Pf
_orig = _m.make_rhs
def make_rhs_noisy(p, Ps_fun, x0_fun):
    r = _orig(p, Ps_fun, x0_fun); nf = _NOISE.get('f'); g = p['lg'] * p['d1'] / p['m1e']
    if nf is None: return r
    def rhs(t, y):
        d = r(t, y); d[1] += Ps_fun(t) * (nf(t) - 1.0) * g; return d
    return rhs
_m.make_rhs = make_rhs_noisy
make_rhs = make_rhs_noisy
def onset_const_n(Ps, T=1.0, seed=7, sigma=SIGMA, **kw):
    p = params(**kw)
    t, y, U, _ = simulate(p, noisy(lambda tt: Ps * min(1.0, tt / 0.005), seed, T, sigma), T, y0=[0, 0, 0, 0] if sigma > 0 else None)
    on, rms = classify(t, U); return on, rms, (f0_of(t, U) if on else np.nan)
def bisect_n(lo=0.0, hi=2000.0, tol=5.0, **kw):
    n = 1
    if not onset_const_n(hi, **kw)[0]: return np.nan, n
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); n += 1
        if onset_const_n(mid, **kw)[0]: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi), n
def onset_time(t, U, Pf):
    tc, r = window_rms(U); above = r > 1e-6
    if not above[-1]: return np.nan, np.nan
    k = len(above) - 1
    while k > 0 and above[k - 1]: k -= 1
    t_on = tc[k] - 0.005; return t_on, Pf(t_on)
