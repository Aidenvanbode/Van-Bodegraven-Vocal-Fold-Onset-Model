"""Step 4c: wide-gap fix, Stage B (Assaneo & Trevisan 2013 viscous loss), contact-parameter checks."""
from tools.runner import main
FULL = [['run_c.py'], ['run_full.py']]
QUICK = """
from model_c import *
for st in 'AB':
    print('Stage', st, '1.3x threshold, 0.4 s (onset, rms, F0):', onset_const(1.3 * (259.8 if st == 'A' else 195.3), T=0.4, stage=st))
    print('Stage', st, '0.7x threshold, 0.4 s (onset, rms, F0):', onset_const(0.7 * (259.8 if st == 'A' else 195.3), T=0.4, stage=st))
"""
if __name__ == '__main__':
    main('step4c', FULL, [['analysis_c.py']], QUICK, __doc__)
