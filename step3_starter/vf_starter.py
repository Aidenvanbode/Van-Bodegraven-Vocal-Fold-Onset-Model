"""
vf_starter.py  --  STARTER sketch of a symmetric two-mass vocal fold model.

Model: Ishizaka & Flanagan (1972) two-mass model, symmetric folds, in the
simplified form used by Steinecke & Herzel (1995) / Lucero & Koenig (2005):
  - Bernoulli flow up to the narrowest point (flow separation at min area),
    free jet after it (no pressure recovery), no air viscosity, no vocal tract.
  - cubic tissue springs, extra (cubic) contact spring when a mass closes,
    extra damping during contact (IF72 zeta: open 0.1/0.6, closed 1.1/1.9).
  - hard switches smoothed with tanh (width EPS); report sensitivity to EPS.
Parameters (IF72 "standard" values, converted CGS -> SI), as listed in
Assaneo & Trevisan 2013 (Papers in Physics 5:050004) and the DINAME 2007 paper.
"""
import numpy as np
from scipy.integrate import solve_ivp

# ---------------- parameters (SI) ----------------
P = dict(
    m1=0.125e-3, m2=0.025e-3,          # kg   (0.125 g, 0.025 g)
    k1=80.0, k2=8.0, kc=25.0,          # N/m  (80,000 / 8,000 / 25,000 dyn/cm)
    eta_k=100e4, eta_h=500e4,          # 1/m^2 (100 and 500 cm^-2)
    zeta_open=(0.1, 0.6), zeta_closed=(1.1, 1.9),
    d1=0.25e-2, d2=0.05e-2, lg=1.4e-2, # m
    x0=0.02e-2,                        # m, rest half-width (0.02 cm)
    rho=1.2,                           # kg/m^3 air (approx.)
    EPS=1e-6,                          # m, smoothing width for switches (CHECK)
)

def H(z, eps):            # smooth Heaviside, ~1 for z>>eps
    return 0.5 * (1.0 + np.tanh(z / eps))

def rhs(t, y, p, Ps_fun):
    x1, v1, x2, v2 = y
    Ps = Ps_fun(t)
    lg, x0, eps = p['lg'], p['x0'], p['EPS']
    a1 = 2 * lg * (x0 + x1); a2 = 2 * lg * (x0 + x2)
    c1 = H(-(x0 + x1), eps); c2 = H(-(x0 + x2), eps)       # contact indicators
    o1 = 1 - c1; o2 = 1 - c2                                # open indicators
    # aerodynamic forces (IF72 table): both open -> Bernoulli with separation
    # at the narrower section; mass 1 closed -> f1 = Ps*lg*d1, f2 = 0;
    # only mass 2 closed -> f1 = f2 = Ps*lg*d*... (both see Ps)
    a1p = max(a1, 1e-12); a2p = max(a2, 1e-12)
    amin = min(a1p, a2p)
    f1_open = Ps * lg * p['d1'] * (1 - (amin / a1p) ** 2)
    f1 = o1 * o2 * f1_open + c1 * Ps * lg * p['d1'] + o1 * c2 * Ps * lg * p['d1']
    f2 = o1 * c2 * Ps * lg * p['d2']
    # springs
    s1 = p['k1'] * x1 * (1 + p['eta_k'] * x1**2) + c1 * 3 * p['k1'] * (x1 + x0) * (1 + p['eta_h'] * (x1 + x0)**2)
    s2 = p['k2'] * x2 * (1 + p['eta_k'] * x2**2) + c2 * 3 * p['k2'] * (x2 + x0) * (1 + p['eta_h'] * (x2 + x0)**2)
    z1 = o1 * p['zeta_open'][0] + c1 * p['zeta_closed'][0]
    z2 = o2 * p['zeta_open'][1] + c2 * p['zeta_closed'][1]
    r1 = 2 * z1 * np.sqrt(p['m1'] * p['k1']); r2 = 2 * z2 * np.sqrt(p['m2'] * p['k2'])
    dv1 = (f1 - s1 - r1 * v1 - p['kc'] * (x1 - x2)) / p['m1']
    dv2 = (f2 - s2 - r2 * v2 - p['kc'] * (x2 - x1)) / p['m2']
    return [v1, dv1, v2, dv2]

def flow(y, Ps, p):
    a1 = 2 * p['lg'] * (p['x0'] + y[0]); a2 = 2 * p['lg'] * (p['x0'] + y[2])
    amin = np.clip(np.minimum(a1, a2), 0, None)
    return amin * np.sqrt(2 * np.clip(Ps, 0, None) / p['rho'])   # m^3/s

def simulate(Ps_fun, T, p=P, rtol=1e-8, atol=1e-11, max_step=2e-5):
    y0 = [1e-6, 0, 1e-6, 0]   # tiny perturbation (1 micrometre) to seed oscillation
    sol = solve_ivp(rhs, (0, T), y0, args=(p, Ps_fun), method='LSODA',
                    rtol=rtol, atol=atol, max_step=max_step, dense_output=False,
                    t_eval=np.arange(0, T, 1 / 44100))
    U = flow(sol.y, np.array([Ps_fun(t) for t in sol.t]), p)
    return sol.t, sol.y, U

# ---------------- check 1: natural frequencies (no flow) ----------------
def natural_freqs(p=P):
    M = np.diag([p['m1'], p['m2']])
    K = np.array([[p['k1'] + p['kc'], -p['kc']], [-p['kc'], p['k2'] + p['kc']]])
    w2 = np.linalg.eigvals(np.linalg.solve(M, K))
    return np.sort(np.sqrt(w2.real)) / (2 * np.pi)

# ---------------- check 2: linear (Hopf) threshold, Lucero & Koenig 2005 eqs 2-4 ----------------
def linear_threshold(p=P):
    m1, m2, k1, k2, kc = p['m1'], p['m2'], p['k1'], p['k2'], p['kc']
    r1 = 2 * p['zeta_open'][0] * np.sqrt(m1 * k1); r2 = 2 * p['zeta_open'][1] * np.sqrt(m2 * k2)
    def rh(G):
        mu1, mu2 = r1 / m1, r2 / m2
        w1, w2 = (k1 + kc - G) / m1, (k2 + kc) / m2
        kap = kc * (kc - G) / (m1 * m2)
        a1, a2, a3, a4 = mu1 + mu2, w1 + w2 + mu1 * mu2, mu1 * w2 + mu2 * w1, w1 * w2 - kap
        return a1 * a2 * a3 - a3**2 - a1**2 * a4
    from scipy.optimize import brentq
    G = np.linspace(0.1, 200, 4000); v = [rh(g) for g in G]
    i = next(i for i in range(1, len(G)) if np.sign(v[i]) != np.sign(v[i - 1]))
    Gth = brentq(rh, G[i - 1], G[i])
    Ps_th = Gth * p['x0'] / (2 * p['d1'] * p['lg'])
    return Gth, Ps_th

def onset_detect(t, U, t_keep=0.2, thr=1e-6):
    """Lucero-style: AC flow RMS > 1 cm^3/s, evaluated on the last t_keep seconds
    (transient discarded) + second check: amplitude not decaying."""
    m = t >= t[-1] - t_keep
    u = U[m] - U[m].mean(); rms = np.sqrt(np.mean(u**2))
    half = len(u) // 2
    r1 = np.sqrt(np.mean(u[:half]**2)) + 1e-30; r2 = np.sqrt(np.mean(u[half:]**2))
    return rms > thr and r2 / r1 > 0.9, rms

if __name__ == '__main__':
    print('Natural frequencies (Hz):', np.round(natural_freqs(), 1))
    Gth, Ps_lin = linear_threshold()
    print(f'Linear Hopf threshold: Gamma_th = {Gth:.2f} N/m  (Lucero&Koenig 2005 report 47.80)')
    print(f'  -> Ps_th/x0 = {Gth/(2*P["d1"]*P["lg"]):.3g} N/m^3 (LK2005: 6.83e5);  Ps_th = {Ps_lin:.1f} Pa at x0=0.02 cm')
    print('\nNonlinear simulations, constant Ps (1.0 s each, last 0.2 s analysed):')
    for Ps in [100, 150, 200, 225, 250, 275, 300, 500, 800]:
        t, y, U = simulate(lambda tt, Ps=Ps: Ps * min(1.0, tt / 0.005), 1.0)
        on, rms = onset_detect(t, U)
        f0 = np.nan
        if on:
            u = U[t >= 0.8] - U[t >= 0.8].mean()
            zc = np.where((u[:-1] < 0) & (u[1:] >= 0))[0]
            if len(zc) > 2: f0 = 44100 / np.mean(np.diff(zc))
        print(f'  Ps={Ps:4d} Pa  onset={on!s:5}  AC-flow RMS={rms*1e6:8.2f} cm^3/s  F0={f0:6.1f} Hz')
    # hysteresis: sinusoidal ramp 0 -> 1000 -> 0 Pa over 1 s (Lucero & Koenig 2005 protocol)
    print('\nHysteresis ramp (0->1000->0 Pa, 1 s):')
    Pfun = lambda tt: 1000 * np.sin(np.pi * tt / 1.0) ** 2 if tt < 1 else 0.0
    t, y, U = simulate(Pfun, 1.0)
    per = 1 / 120; n = int(per * 44100)
    def ac_rms(seg):   # remove linear trend (slowly varying Ps changes the DC flow)
        tt = np.arange(len(seg)); c = np.polyfit(tt, seg, 1)
        return np.std(seg - np.polyval(c, tt))
    rms = np.array([ac_rms(U[i:i + n]) for i in range(0, len(U) - n, n)])
    tc = np.arange(0, len(U) - n, n) / 44100 + per / 2
    above = rms > 1e-6
    if above.any():
        i_on = np.argmax(above); i_off = len(above) - 1 - np.argmax(above[::-1])
        print(f'  onset at t={tc[i_on]:.3f}s Ps={Pfun(tc[i_on]):.0f} Pa; offset at t={tc[i_off]:.3f}s Ps={Pfun(tc[i_off]):.0f} Pa;'
              f' offset/onset ratio={Pfun(tc[i_off])/Pfun(tc[i_on]):.2f}')
    else:
        print('  no oscillation detected')
