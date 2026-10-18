"""Step 3: starter model (first sketch of the two-mass model; checks linear threshold vs Lucero & Koenig 2005)."""
from tools.runner import main
QUICK = """
import numpy as np, vf_starter as v
print('natural freqs (Hz)', np.round(v.natural_freqs(), 1))
G, P = v.linear_threshold(); print('linear threshold Gamma_th', round(G, 2), 'N/m (LK2005: 47.80); Ps_th', round(P, 1), 'Pa')
t, y, U = v.simulate(lambda tt: 800 * min(1.0, tt / 0.005), 0.3); print('800 Pa, 0.3 s: onset', v.onset_detect(t, U, t_keep=0.1))
"""
if __name__ == '__main__':
    main('step3_starter', full_cmds=[['vf_starter.py']], analysis_cmds=[], quick_code=QUICK, description=__doc__)
