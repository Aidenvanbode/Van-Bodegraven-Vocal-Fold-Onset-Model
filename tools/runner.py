"""Shared helpers for the run_step*.py scripts (standard library only)."""
import gzip, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unpack_data():
    """Decompress the *.gz result files (stored compressed to keep the repo small) next to themselves, if missing."""
    for d, _, files in os.walk(ROOT):
        if '.git' in d:
            continue
        for f in files:
            if f.endswith('.gz'):
                src = os.path.join(d, f); dst = src[:-3]
                if not os.path.exists(dst):
                    with gzip.open(src, 'rb') as a, open(dst, 'wb') as b:
                        shutil.copyfileobj(a, b)


def run(script_args, step_dir, cwd_root=ROOT):
    """Run `python <script> args...` inside <cwd_root>/<step_dir> (scripts use paths relative to their own folder)."""
    cwd = os.path.join(cwd_root, step_dir)
    print(f'\n>>> [{step_dir}] python {" ".join(script_args)}', flush=True)
    t0 = time.time()
    subprocess.run([sys.executable] + list(script_args), cwd=cwd, check=True)
    print(f'<<< done in {time.time() - t0:.1f} s', flush=True)


def temp_copy():
    """Copy the whole repo to a temporary folder, so quick tests never overwrite the stored results."""
    tmp = tempfile.mkdtemp(prefix='vf_quick_')
    dst = os.path.join(tmp, 'repo')
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.gz'))
    return dst


def quick_python(code, step_dir, cwd_root):
    """Run a short Python snippet inside a step folder (so the step's own modules import normally)."""
    print(f'\n>>> [{step_dir}] quick model check', flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, '-c', code], cwd=os.path.join(cwd_root, step_dir), check=True)
    print(f'<<< done in {time.time() - t0:.1f} s', flush=True)


def main(step_dir, full_cmds, analysis_cmds, quick_code, description):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--quick', action='store_true', help='tiny smoke test (< 2 min) in a temporary copy; stored results untouched')
    g.add_argument('--analysis', action='store_true', help='re-make the tables/figures of this step from the stored result files')
    g.add_argument('--full', action='store_true', help='re-run every simulation of this step (slow: hours), then the analysis')
    a = ap.parse_args()
    unpack_data()
    if a.quick:
        t0 = time.time(); root = temp_copy(); print('quick test in', root)
        quick_python(quick_code, step_dir, root)
        for c in analysis_cmds:
            run(c, step_dir, root)
        shutil.rmtree(os.path.dirname(root), ignore_errors=True)
        print(f'\nQUICK TEST PASSED ({time.time() - t0:.1f} s)')
    elif a.full:
        print('FULL RUN: this re-runs every simulation for this step and overwrites its result files. It can take hours.')
        for c in full_cmds:
            run(c, step_dir)
        for c in analysis_cmds:
            run(c, step_dir)
    elif a.analysis:
        for c in analysis_cmds:
            run(c, step_dir)
    else:
        ap.print_help()
