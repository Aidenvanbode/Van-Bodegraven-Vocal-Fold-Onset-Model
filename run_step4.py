"""Step 4: first experiment run, Stage A, no noise (superseded by 4b-4f; kept because the report cites it)."""
from tools.runner import main
FULL = [['part1.py'], ['part1_hyst.py'], ['part2_validation.py'], ['exp1.py'], ['exp23.py', '2'], ['exp23.py', '3', '100']]
QUICK = """
from model import *
print('linear threshold (Gamma_th, Ps_th):', linear_threshold())
print('800 Pa, 0.4 s (onset, AC rms, F0):', onset_const(800, T=0.4))
print('100 Pa, 0.4 s (onset, AC rms, F0):', onset_const(100, T=0.4))
"""
if __name__ == '__main__':
    main('step4', FULL, [['analysis.py']], QUICK, __doc__)
