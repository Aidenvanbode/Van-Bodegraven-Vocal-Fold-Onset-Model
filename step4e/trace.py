import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from model_e import *
fig, axs = plt.subplots(4, 3, figsize=(13, 9), sharex='col')
cases = [('B', 'easy', None), ('B', 'hard', 0.020), ('B', 'hard', 0.100)]
for col, (st, start, Trel) in enumerate(cases):
    p = params(stage=st); x0 = 0.2e-3; Pt = 1.2 * 195.3; Tr = 0.2
    Pf = lambda t: Pt * min(t / Tr, 1)
    w = w_release(Tr, Trel) if start == 'hard' else None
    t, y, U, Ps = simulate(p, Pf, 0.6, x0, w, seed=5000)
    o = onset_growth(t, U, Pf); tc, A, M, R = cycles(t, U)
    axs[0, col].plot(t*1e3, Ps); axs[0, col].set_ylabel('Ps (Pa)'); axs[0, col].set_title(f'Stage {st} {start}' + (f', release {Trel*1e3:.0f} ms' if Trel else ''))
    axs[1, col].plot(t*1e3, (x0+y[0])*1e3, label='mass 1'); axs[1, col].plot(t*1e3, (x0+y[2])*1e3, label='mass 2'); axs[1, col].axhline(0, color='k', lw=.5); axs[1, col].set_ylabel('half-gap (mm)'); axs[1, col].legend(fontsize=7)
    axs[2, col].plot(t*1e3, U*1e6); axs[2, col].set_ylabel('flow (cm³/s)')
    if w: axs[3, col].plot(t*1e3, [w(s) for s in t], 'k:', label='hold w(t)')
    if len(R): axs[3, col].semilogy(tc*1e3, np.maximum(R, 1e-4), '.', label='cycle R = A/mean')
    axs[3, col].axhline(0.1, color='r', lw=.7); axs[3, col].set_ylabel('R / hold'); axs[3, col].set_xlabel('time (ms)'); axs[3, col].legend(fontsize=7)
    for a in axs[:, col]:
        if np.isfinite(o['t_on']): a.axvline(o['t_on']*1e3, color='m', ls='--')
    print(st, start, Trel, o, 'max opening while held (mm):', ((x0+np.minimum(y[0], y[2]))[t < Tr]).max()*1e3)
fig.tight_layout(); fig.savefig('fig1_traces.png', dpi=130)
