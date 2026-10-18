"""Step 4f model layer = Step 4e model (model_e) + (1) fixed onset detector, (2) displacement-based detector,
(3) hold stiffness / hold position as per-call settings, (4) beta (size) scaling and wide parameter draws."""
import sys; sys.path.insert(0, '../step4e'); sys.path.insert(0, '../step4c')
import numpy as np
from scipy.signal import find_peaks
import model_e as me
from model_e import BASE, FS, params, w_release, cycles
import model_c as mc
def set_hold(k_add=2000.0, x_ad=-0.05e-3):
    me.K_ADD = float(k_add); me.X_AD = float(x_ad)
def simulate(*a, **k): return me.simulate(*a, **k)
def _decay_tests(tc_e, V, keep):
    """Two 'not decaying' tests on the end-window cycle series V (R or displacement amplitude).
    v1 (Step 4e): P90 of 2nd half >= 0.9 x P90 of 1st half.
    trend (Step 4f): envelope E_k = max(V_k, V_k+1) (removes the alternation from two flow peaks per period);
    least-squares slope of log E vs time; passes if the fitted change across the window exp(slope*keep) >= 0.9."""
    h = len(V) // 2; e1, e2 = np.percentile(V[:h], 90), np.percentile(V[h:], 90)
    v1 = bool(e2 >= 0.9 * e1)
    E = np.maximum(V[:-1], V[1:]); tt = tc_e[:-1]
    sl = np.polyfit(tt, np.log(np.maximum(E, 1e-12)), 1)[0] if len(E) >= 3 else 0.0
    return v1, bool(np.exp(sl * keep) >= 0.9), float(sl)
def _onset(tc, V, Vstar, Pf, t_from):
    sel = tc >= t_from; idx = np.where(sel)[0]
    if len(idx) == 0: return np.nan, np.nan, np.nan
    ab = V[idx] >= Vstar; j = len(idx) - 1
    while j > 0 and ab[j - 1]: j -= 1
    i1 = idx[j]
    if j == 0: ton = tc[i1]; grew = 0.0
    else:
        i0 = idx[j - 1]; l0, l1, ls = np.log(max(V[i0], 1e-12)), np.log(V[i1]), np.log(Vstar)
        ton = tc[i0] + (tc[i1] - tc[i0]) * (ls - l0) / (l1 - l0) if l1 != l0 else tc[i1]
        k0 = max(0, j - 5); lr = np.log(np.maximum(V[idx[k0:j + 1]], 1e-12))
        grew = float(np.polyfit(tc[idx[k0:j + 1]], lr, 1)[0] > 0) if j - k0 >= 2 else 1.0
    return float(ton), float(Pf(ton)), grew
def detect_series(t, tc, V, Pf, stars, t_from=0.0, keep=0.2, tag=''):
    """Apply both detector versions for every cut-off in `stars`. Columns: succ_v1_<tag>, succ_<tag> (v2 = fixed),
    t_on_<tag>, Ps_on_<tag>, grew_<tag>, plus end_<tag> (P90 of V in end window). v2 success = level >= cut-off AND
    (v1 decay test OR trend test). v2 is a superset of v1 by construction: it can only rescue v1 false negatives."""
    out = {}
    ok = len(V) >= 6
    if ok:
        end = tc >= t[-1] - keep; ok = end.sum() >= 4
    lvl = float(np.percentile(V[end], 90)) if ok else np.nan
    if ok: v1d, trd, sl = _decay_tests(tc[end], V[end], keep)
    else: v1d = trd = False; sl = np.nan
    out[f'end{tag}'] = lvl; out[f'slope{tag}'] = sl; out[f'v1dec{tag}'] = v1d; out[f'trdec{tag}'] = trd
    for s in stars:
        k = f'{tag}{s:g}'
        s1 = bool(ok and lvl >= s and v1d); s2 = bool(ok and lvl >= s and (v1d or trd))
        out[f'succ_v1_{k}'] = s1; out[f'succ_{k}'] = s2
        ton, pon, gr = _onset(tc, V, s, Pf, t_from) if s2 else (np.nan, np.nan, np.nan)
        out[f't_on_{k}'] = ton; out[f'Ps_on_{k}'] = pon; out[f'grew_{k}'] = gr
    return out
def xcycles(t, X):
    """Displacement cycles: peaks of the mass-1 opening x0+x1 (>= 2 ms apart); per cycle A = peak - min (m)."""
    pk, _ = find_peaks(X, distance=int(0.002 * FS))
    if len(pk) < 3: return np.array([]), np.array([])
    tc = t[pk[:-1]]; A = np.array([X[a] - X[a:b + 1].min() for a, b in zip(pk[:-1], pk[1:])]); return tc, A
RSTARS = (0.05, 0.10, 0.20)
AX_STAR = 0.02e-3   # displacement criterion: 0.02 mm peak-to-trough of mass 1 (10% of the 0.2 mm standard rest gap)
def detect_all(t, y, U, x0, Pf, t_from=0.0, keep=0.2, rstars=RSTARS):
    tc, A, M, R = cycles(t, U); o = detect_series(t, tc, R, Pf, rstars, t_from, keep, 'R')
    if len(tc) >= 3: o['F0'] = float(1 / np.median(np.diff(tc[tc >= t[-1] - keep]))) if (tc >= t[-1] - keep).sum() > 1 else np.nan
    txc, Ax = xcycles(t, x0 + y[0]); o.update(detect_series(t, txc, Ax, Pf, (AX_STAR,), t_from, keep, 'X'))
    return o
# ---- size scaling (Lucero & Koenig 2005, JASA 118:2798): all dimensions x beta, masses x beta^3, stiffness x beta,
# damping ratio constant (so damping coefficient x beta^2 automatically). Rest gap x0 is a control parameter, not scaled.
# Nonlinear spring coefficients eta (1/m^2) scaled by beta^-2 so the nonlinearity acts at the same *relative* displacement (our assumption).
def beta_kw(beta):
    b = beta
    return dict(m1=BASE['m1'] * b**3, m2=BASE['m2'] * b**3, k1=BASE['k1'] * b, k2=BASE['k2'] * b, kc=BASE['kc'] * b,
                d1=BASE['d1'] * b, d2=BASE['d2'] * b, lg=BASE['lg'] * b, eta_k=BASE['eta_k'] / b**2, eta_h=BASE['eta_h'] / b**2)
def linear_threshold_p(p, x0):
    """Lucero & Koenig 2005 linear stability (no viscosity, no tract): Gamma_th from Routh-Hurwitz; Ps = Gamma x0 / (2 d1 lg)."""
    from scipy.optimize import brentq
    m1, m2, k1, k2, kc = p['m1e'], p['m2e'], p['k1e'], p['k2e'], p['kce']
    r1 = 2 * p['zo'][0] * np.sqrt(m1 * k1); r2 = 2 * p['zo'][1] * np.sqrt(m2 * k2)
    def rh(G):
        mu1, mu2 = r1 / m1, r2 / m2; w1, w2 = (k1 + kc - G) / m1, (k2 + kc) / m2
        kap = kc * (kc - G) / (m1 * m2)
        a1, a2, a3, a4 = mu1 + mu2, w1 + w2 + mu1 * mu2, mu1 * w2 + mu2 * w1, w1 * w2 - kap
        return a1 * a2 * a3 - a3 ** 2 - a1 ** 2 * a4
    G = np.linspace(0.01, 500, 50000); v = np.array([rh(g) for g in G])
    i = np.where(np.sign(v[1:]) != np.sign(v[:-1]))[0][0]; Gth = brentq(rh, G[i], G[i + 1])
    return Gth, Gth * x0 / (2 * p['d1'] * p['lg'])
