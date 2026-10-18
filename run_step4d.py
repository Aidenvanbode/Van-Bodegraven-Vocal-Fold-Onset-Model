"""Step 4d: tests of the H2 artifact (window mid-point, blow-open phase, no-ramp kick, AC/DC cut-off table)."""
from tools.runner import main
FULL = [['run4d.py'], ['diag.py']]
QUICK = """
import run4d as r
o = r.kick_run(('B', 'hard', 1.2, 0, 0.020)); print('kick run, Stage B, hard, 1.2x, 20 ms release:', {k: o[k] for k in list(o)[:8]})
"""
if __name__ == '__main__':
    main('step4d', FULL, [['map_cutoffs.py'], ['analysis4d.py']], QUICK, __doc__)
