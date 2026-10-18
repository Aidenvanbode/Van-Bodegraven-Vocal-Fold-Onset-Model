"""Step 5: figures for the reference draft (5c figures and the updated 5d figures), from stored result files."""
import os
from tools.runner import main
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'step5_figs', 'figs'), exist_ok=True)
QUICK = "print('figure scripts only read stored CSV files; no simulation needed')"
if __name__ == '__main__':
    main('step5_figs', [], [['make_figs.py'], ['make_figs_5d.py']], QUICK, __doc__)
