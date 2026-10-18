"""Step 4f: detector fix, robustness (hold stiffness/position, R cut-off), race on random draws, V3 and V7 checks.
Note: 4f was run at reduced scale (see README). run4f.py is resumable and skips jobs already in results.jsonl;
delete step4f/results.jsonl first to recompute everything from scratch."""
from tools.runner import main
FULL = [['run4f.py']]
QUICK = """
import run4f as r, model_f as mf
for start, rel in (('easy', 0.0), ('hard', 0.005), ('hard', 0.100)):
    o = r.const_run('B', start, 1.2 * r.PTH['B'], 0, rel, 3000, Tafter=0.3)
    print(f'Stage B 1.2x {start:4s} release {rel*1e3:5.0f} ms: success {o["succ_R0.1"]}, delay {o["delay_ms_R0.1"]:.1f} ms')
G, P = mf.linear_threshold_p(mf.params(stage='B'), r.XT); print('linear threshold Gamma_th', round(G, 2), 'N/m (LK2005 eq. 5: 47.80), Ps_lin', round(P, 1), 'Pa')
"""
if __name__ == '__main__':
    main('step4f', FULL, [['analysis4f.py']], QUICK, __doc__)
