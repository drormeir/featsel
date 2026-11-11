"""
Run an experiment to the end, resuming it after a crash.

Compiled code such as liblinear aborts when memory is refused, instead of
raising MemoryError. With several workers the runner survives this: it halves
its workers and resumes. With one worker, or when the crash hits the runner's
own process, the run stops. The runner saves every row as it finishes, so a
new process with --resume continues where the crash stopped.

This script starts the runner in its own process and, after a crash, starts
it again with --resume, up to --attempts times. Every other argument goes to
the runner unchanged.

Usage:
    python scripts/run_until_done.py --attempts 3 --config configs/<name>.yaml --n-jobs 4
"""

import argparse
import subprocess
import sys


def run_until_done(runner_args, attempts, runner=('-m', 'featsel.run')):
    """Run the runner until it succeeds or attempts run out; return its last exit code."""
    command = [sys.executable, *runner, *runner_args]
    if '--resume' not in runner_args:
        command.append('--resume')
    for attempt in range(1, attempts + 1):
        code = subprocess.run(command, check=False).returncode
        if code == 0:
            return 0
        print(f'\nThe runner stopped with exit code {code} (attempt {attempt} of {attempts}).',
              flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--attempts', type=int, default=3,
                        help='Runner starts allowed, the first included')
    args, runner_args = parser.parse_known_args()
    sys.exit(run_until_done(runner_args, args.attempts))


if __name__ == '__main__':
    main()
