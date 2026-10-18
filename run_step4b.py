"""Step 4b: rerun with 0.5% subglottal-pressure noise and a fair easy/hard comparison (Stage A)."""
from tools.runner import main
FULL = [['part1.py'], ['exps.py', '200']]
QUICK = """
from common import *
print('800 Pa with 0.5% noise, 0.4 s:', onset_const_n(800, T=0.4))
print('150 Pa with 0.5% noise, 0.4 s:', onset_const_n(150, T=0.4))
"""
if __name__ == '__main__':
    main('step4b', FULL, [['analysis.py']], QUICK, __doc__)
