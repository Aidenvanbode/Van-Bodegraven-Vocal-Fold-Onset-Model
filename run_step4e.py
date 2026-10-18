"""Step 4e: adductory hold spring, cycle-by-cycle onset detector, thresholds / delays / paths / H4 / onset maps."""
from tools.runner import main
FULL = [['run4e.py'], ['trace.py']]
QUICK = """
import run4e as r
for start, rel in (('easy', 0.0), ('hard', 0.005), ('hard', 0.100)):
    o = r.const_run('B', start, 1.2 * r.PTH['B'], 0, rel, 3000, Tafter=0.3)
    print(f'Stage B 1.2x {start:4s} release {rel*1e3:5.0f} ms: success {o["success"]}, delay {o["delay_ms"]:.1f} ms')
"""
if __name__ == '__main__':
    main('step4e', FULL, [['analysis4e.py']], QUICK, __doc__)
