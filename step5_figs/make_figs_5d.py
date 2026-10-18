"""Figure 1 of the step 5d reference draft: model schematic with the adductory hold, and the easy / hard paths
(pressure ramp, hold weight w(t) and its raised-cosine release over 5, 20 or 100 ms). Pure drawing, no simulation.
Output: figs/fig1_model_paths_5d.png (300 dpi)."""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
O = sys.argv[1] if len(sys.argv) > 1 else 'figs/'
os.makedirs(O, exist_ok=True)
EASY = '#1f77b4'; HARD = '#d62728'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'savefig.dpi': 300})
fig = plt.figure(figsize=(7.4, 3.2)); ax = fig.add_axes([0.0, 0.02, 0.34, 0.96]); ax.set_xlim(0, 10); ax.set_ylim(-1.3, 10); ax.axis('off')
for s in (-1, 1):
    cx = 5 + s * 1.3
    for (y, h, lab, w) in ((2.2, 2.6, 'm$_1$', 1.3), (5.4, 1.4, 'm$_2$', 1.0)):
        x = cx - (0 if s > 0 else w); ax.add_patch(Rectangle((x, y), w, h, fc='#cccccc', ec='k'))
        ax.text(x + w / 2, y + h / 2, lab, ha='center', va='center')
        xs = cx + (w if s > 0 else -w); xe = xs + s * 1.6
        zz = np.linspace(xs, xe, 40); ax.plot(zz, y + h / 2 + 0.25 * np.sin(np.linspace(0, 8 * np.pi, 40)), 'k', lw=.7)
        ax.plot([xe, xe], [y + h / 2 - 0.5, y + h / 2 + 0.5], 'k', lw=1.5)
    ax.plot([cx + s * 0.5] * 2, [4.8, 5.4], 'k', lw=.7)
    ax.annotate('', xy=(cx + s * 0.1, 7.6), xytext=(cx + s * 1.6, 7.6), arrowprops=dict(arrowstyle='->', color=HARD, lw=1.2))
ax.text(5, 8.0, 'hold force w(t)K$_{add}$ (hard only)', ha='center', fontsize=6.5, color=HARD)
ax.text(5.0, 4.9, 'k$_c$', fontsize=7, ha='center')
ax.annotate('', xy=(5, 1.6), xytext=(5, 0.2), arrowprops=dict(arrowstyle='->'))
ax.text(5, 0.0, 'P$_s$ (subglottal)', ha='center', fontsize=8, va='top')
ax.annotate('', xy=(5, 9.6), xytext=(5, 8.6), arrowprops=dict(arrowstyle='->')); ax.text(5.3, 9.3, 'U$_g$', fontsize=8)
ax.annotate('', xy=(3.7, 1.8), xytext=(6.3, 1.8), arrowprops=dict(arrowstyle='<->', lw=.7)); ax.text(5, 1.75, '2x$_0$', ha='center', va='top', fontsize=7)
ax.text(0.2, 9.6, '(a)', fontweight='bold')
Tr = 0.2; t = np.linspace(0, 0.45, 2000)
def w(t, t0, T):
    return np.where(t < t0, 1.0, np.where(t < t0 + T, 0.5 * (1 + np.cos(np.pi * (t - t0) / T)), 0.0))
a1 = fig.add_axes([0.43, 0.58, 0.53, 0.33]); a2 = fig.add_axes([0.43, 0.14, 0.53, 0.33])
a1.plot(t * 1e3, 1.2 * np.minimum(t / Tr, 1), 'k'); a1.set_ylabel('P$_s$ / P$_{s,th}$'); a1.set_xticklabels([])
a1.set_title('(b) 200 ms ramp to 1.2 x threshold; hard path held until the ramp ends')
for T, ls in ((0.005, '-'), (0.020, '--'), (0.100, ':')):
    a2.plot(t * 1e3, w(t, Tr, T), ls, color=HARD, label=f'hard, release {T*1e3:.0f} ms')
a2.plot(t * 1e3, 0 * t, '-', color=EASY, lw=2, alpha=.6, label='easy (no hold)')
a2.set_xlabel('time (ms)'); a2.set_ylabel('hold weight w(t)'); a2.set_ylim(-0.1, 1.15); a2.legend(fontsize=6.5, loc='center left')
fig.savefig(os.path.join(O, 'fig1_model_paths_5d.png')); plt.close()
print('wrote', os.path.join(O, 'fig1_model_paths_5d.png'))
