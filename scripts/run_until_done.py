"""
Run an experiment to the end, resuming it after a crash.

Compiled code such as liblinear aborts when memory is refused, instead of
raising MemoryError. With several workers the runner survives this: it halves
its workers and resumes. With one worker the job runs in the runner's own
process, so the crash ends the run. The runner saves every row as it
finishes, and notes the job that was running, so a new process with --resume
continues where the crash stopped and skips that job (featsel.run).

This script starts the runner in its own process and, after a crash, starts
it again with --resume and the number of workers the runner had fallen back
to. It stops when the run completes, or after --attempts restarts in a row
that saved no new row and recorded no new crashed job. While it runs, the
machine is kept from sleeping (systemd-inhibit, where it exists). Every
other argument goes to the runner unchanged.

Usage:
    python scripts/run_until_done.py --config configs/<name>.yaml --n-jobs 4
"""

import argparse
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from featsel.run import load_config


def run_until_done(runner_args, attempts, runner=('-m', 'featsel.run')):
    """Run the runner until it succeeds or stops progressing; return its last exit code."""
    args = list(runner_args)
    if '--resume' not in args:
        args.append('--resume')
    output = _output(args)
    with keep_awake():
        return _restart_loop(args, attempts, runner, output)


@contextmanager
def keep_awake():
    """Keep the machine from sleeping inside the block, where systemd-inhibit exists."""
    if shutil.which('systemd-inhibit') is None:
        yield
        return
    inhibitor = subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--who=featsel',
                                  '--why=an experiment is running', 'sleep', 'infinity'])
    try:
        yield
    finally:
        inhibitor.terminate()
        inhibitor.wait()


def _restart_loop(args, attempts, runner, output):
    """Start the runner, and restart it after each crash while it progresses."""
    stalled = 0
    progress = _progress(output)
    while True:
        code = subprocess.run([sys.executable, *runner, *args], check=False).returncode
        if code == 0:
            return 0
        now = _progress(output)
        stalled = 0 if now != progress else stalled + 1
        progress = now
        print(f'\nThe runner stopped with exit code {code}.', flush=True)
        if stalled >= attempts:
            print(f'{attempts} restarts in a row made no progress; giving up.', flush=True)
            return code
        n_jobs = _workers_in_use(output)
        if n_jobs is not None:
            args = _with_option(args, '--n-jobs', str(n_jobs))
        print('Restarting with --resume' + (f' and {n_jobs} workers.' if n_jobs else '.'),
              flush=True)


def _output(args):
    """The output CSV the runner writes, from --out or the config; None if unknown."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config')
    parser.add_argument('--out')
    known, _ = parser.parse_known_args(args)
    if known.out:
        return Path(known.out)
    if known.config:
        return Path(load_config(known.config)['output'])
    return None


def _progress(output):
    """What a run leaves behind: the output's size and the jobs that crashed."""
    if output is None:
        return None
    crashed = output.with_suffix('.crashed')
    return (output.stat().st_size if output.exists() else 0,
            crashed.read_text() if crashed.exists() else '')


def _workers_in_use(output):
    """The workers the last run ended with, from its JSON; None if unknown."""
    if output is None or not output.with_suffix('.json').exists():
        return None
    return json.loads(output.with_suffix('.json').read_text())['config']['n_jobs']


def _with_option(args, option, value):
    """args with option set to value, replacing any earlier value."""
    if option in args:
        i = args.index(option)
        return [*args[:i], option, value, *args[i + 2:]]
    return [*args, option, value]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--attempts', type=int, default=3,
                        help='Restarts in a row without progress before giving up')
    args, runner_args = parser.parse_known_args()
    sys.exit(run_until_done(runner_args, args.attempts))


if __name__ == '__main__':
    main()
