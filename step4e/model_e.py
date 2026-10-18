"""Step 4e model = Step 4c model_c (Stage A/B, IF72 mechanics, 0.5% Ps noise force on mass 1, fixed seeds)
+ ACTIVE ADDUCTORY HOLD: extra force on each mass  F_i = -w(t) * K_ADD * (x0 + x_i - X_AD),
pulling the fold surface toward X_AD = -0.05 mm (past the midline). w(t)=1 while held, falls to 0 over T_rel.
Tissue rest gap x0 stays at the easy-path value (0.2 mm * draw factor), so easy and hard differ ONLY by the hold.
+ GROWTH-BASED ONSET: cycle-by-cycle flow amplitude from peaks (no fixed windows)."""
import sys; sys.path.insert(0, '../step4c')
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import model_c as mc
from model_c import BASE, FS, params, H, aero_B, noise_fun
K_ADD = 2000.0      # N/m per mass (25x k1). Static blow-open pressure with hold ~ K_ADD*0.05mm/(lg*d1) ~ 2.9 kPa
X_AD = -0.05e-3
def make_rhs(p, Ps_fun, x0, w_fun, nf):
    base = mc.make_rhs(p, Ps_fun, lambda t: x0, nf)
    m1, m2 = p['m1e'], p['m2e']
    def rhs(t, y):
        d = base(t, y); w = w_fun(t)
        if w > 0:
            d[1] -= w * K_ADD * (x0 + y[0] - X_AD) / m1; d[3] -= w * K_ADD * (x0 + y[2] - X_AD) / m2
        return d
    return rhs
def w_release(t_start, T_rel):
    """1 until t_start, raised-cosine fall to 0 over T_rel."""
    def w(t):
        if t <= t_start: return 1.0
        if t >= t_start + T_rel: return 0.0
        return 0.5 * (1 + np.cos(np.pi * (t - t_start) / T_rel))
    return w
def simulate(p, Ps_fun, T, x0, w_fun=None, seed=7, sigma=mc.SIGMA, fs=FS):
    held = w_fun is not None
    if not held: w_fun = lambda t: 0.0
    nf = noise_fun(seed, T, sigma) if sigma > 0 else None
    y0 = [X_AD - x0, 0, X_AD - x0, 0] if held else [0, 0, 0, 0]
    te = np.arange(0, T, 1 / fs)
    sol = solve_ivp(make_rhs(p, Ps_fun, x0, w_fun, nf), (0, T), y0, method='LSODA', rtol=1e-8, atol=1e-11, max_step=2e-5, t_eval=te)
    Psv = np.array([Ps_fun(t) for t in sol.t]); a1 = 2 * p['lg'] * (x0 + sol.y[0]); a2 = 2 * p['lg'] * (x0 + sol.y[2])
    if p['stage'] == 'B':
        U = np.array([aero_B(P, A1, A2, p)[2] if min(A1, A2) > 0 else 0.0 for P, A1, A2 in zip(Psv, np.clip(a1, 1e-12, None), np.clip(a2, 1e-12, None))])
    else:
        U = np.clip(np.minimum(a1, a2), 0, None) * np.sqrt(2 * np.clip(Psv, 0, None) / p['rho'])
    return sol.t, sol.y, U, Psv
def cycles(t, U):
    """Cycle-by-cycle: flow peaks (>= 2 ms apart, i.e. F0 <= 500 Hz); per cycle (peak i -> peak i+1):
    A = peak - min between peaks (peak-to-trough), M = mean flow in the cycle, R = A / M."""
    pk, _ = find_peaks(U, distance=int(0.002 * FS))
    if len(pk) < 3: return np.array([]), np.array([]), np.array([]), np.array([])
    tc, A, M = [], [], []
    for a, b in zip(pk[:-1], pk[1:]):
        seg = U[a:b + 1]; tc.append(t[a]); A.append(U[a] - seg.min()); M.append(seg.mean())
    tc, A, M = map(np.array, (tc, A, M)); return tc, A, M, A / np.maximum(M, 1e-12)
def onset_growth(t, U, Pf, Rstar=0.10, t_from=0.0, keep=0.2):
    """success: in the last `keep` s, envelope (90th pct) of cycle R >= Rstar and not decaying (2nd-half envelope >= 0.9 x 1st half).
    onset: start of the final uninterrupted run of cycles with R >= Rstar (after t_from); exact time by log-linear interpolation
    of R between the last sub-threshold cycle and the first supra-threshold cycle. growth flag: log R slope over the
    up-to-5 cycles before onset > 0 (i.e. onset reached by growth, not by a one-off jolt)."""
    tc, A, M, R = cycles(t, U); out = dict(success=False, t_on=np.nan, Ps_on=np.nan, grew=np.nan, R_end=np.nan, F0=np.nan)
    if len(R) < 6: return out
    end = tc >= t[-1] - keep
    if end.sum() < 4: return out
    Re = R[end]; h = len(Re) // 2; out['F0'] = float(1 / np.median(np.diff(tc[end])))
    # envelope = 90th percentile of cycle R (robust when the flow has 2 peaks per period, which makes R alternate)
    e1, e2 = np.percentile(Re[:h], 90), np.percentile(Re[h:], 90); out['R_end'] = float(np.percentile(Re, 90))
    out['success'] = bool(out['R_end'] >= Rstar and e2 >= 0.9 * e1)
    if not out['success']: return out
    sel = tc >= t_from; idx = np.where(sel)[0]; ab = R[idx] >= Rstar
    j = len(idx) - 1
    while j > 0 and ab[j - 1]: j -= 1
    i1 = idx[j]
    if j == 0:  # supra-threshold from the first cycle after t_from: no sub-threshold cycle to interpolate from
        out['t_on'] = tc[i1]; out['grew'] = 0.0
    else:
        i0 = idx[j - 1]; l0, l1, ls = np.log(max(R[i0], 1e-9)), np.log(R[i1]), np.log(Rstar)
        out['t_on'] = tc[i0] + (tc[i1] - tc[i0]) * (ls - l0) / (l1 - l0) if l1 != l0 else tc[i1]
        k0 = max(0, j - 5); lr = np.log(np.maximum(R[idx[k0:j + 1]], 1e-9))
        out['grew'] = float(np.polyfit(tc[idx[k0:j + 1]], lr, 1)[0] > 0) if j - k0 >= 2 else 1.0
    out['Ps_on'] = float(Pf(out['t_on'])); return out
