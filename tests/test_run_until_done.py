"""Tests for scripts/run_until_done.py: resuming the runner after a real crash."""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from featsel.metrics import KEY_COLUMNS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from run_until_done import run_until_done

# A runner that aborts its own process, as liblinear does when memory is
# refused, after its first row the first time it runs.
CRASHING_RUNNER = '''
import os
import sys
from pathlib import Path

import featsel.run as run_module

marker = Path(sys.argv.pop(1))
real_iter_rows = run_module.iter_rows


def crash_once(*args, **kwargs):
    rows = real_iter_rows(*args, **kwargs)
    yield next(rows)
    if not marker.exists():
        marker.touch()
        os.abort()
    yield from rows


run_module.iter_rows = crash_once
run_module.main()
'''


# A runner whose process aborts every time it runs split 1, as liblinear does
# on a cell that never fits in memory.
ALWAYS_CRASHING_JOB = '''
import os

import featsel.run as run_module

real_selection_rows = run_module._selection_rows


def crash_on_split_1(*args, **kwargs):
    if args[4]['split'] == 1:
        os.abort()
    return real_selection_rows(*args, **kwargs)


run_module._selection_rows = crash_on_split_1
run_module.main()
'''


def small_experiment(tmp_path):
    """A one-task experiment on a CSV dataset: 6 executions."""
    X = pd.DataFrame([[float(i * j % 7) for j in range(20)] for i in range(40)],
                     index=[f's{i}' for i in range(40)],
                     columns=[f'g{j}' for j in range(20)])
    X.T.to_csv(tmp_path / 'features.csv')
    pd.DataFrame({'samplename': X.index, 'label': ['a', 'b'] * 20}).to_csv(
        tmp_path / 'metadata.csv', index=False)
    (tmp_path / 'data.yaml').write_text(yaml.safe_dump({
        'name': 'test',
        'paths': {'features': str(tmp_path / 'features.csv'),
                  'metadata': str(tmp_path / 'metadata.csv')},
        'sample_id_column': 'samplename', 'target_column': 'label',
        'transpose_features': True, 'separator': ',',
    }))
    config = tmp_path / 'experiment.yaml'
    config.write_text(yaml.safe_dump({
        'dataset': str(tmp_path / 'data.yaml'), 'output': str(tmp_path / 'out.csv'),
        'n_splits': 3, 'train_sizes': [0.5], 'metrics': ['accuracy'],
        'selection_repeats': 1, 'n_jobs': 1,
        'selectors': [{'name': 'RandomSelector', 'k': [2, 5]}],
        'models': [{'name': 'knn'}],
    }))
    return config


def test_resumes_after_the_runner_process_aborts(tmp_path):
    """Test that a runner killed by abort() is restarted with --resume and completes the grid."""
    config = small_experiment(tmp_path)
    runner = tmp_path / 'crashing_runner.py'
    runner.write_text(CRASHING_RUNNER)
    marker = tmp_path / 'crashed'

    code = run_until_done(['--config', str(config)], attempts=3,
                          runner=(str(runner), str(marker)))

    out = pd.read_csv(tmp_path / 'out.csv')
    assert code == 0
    assert marker.exists()
    assert len(out) == 6
    assert not out.duplicated(KEY_COLUMNS).any()


def test_a_job_that_always_crashes_is_skipped_and_reported(tmp_path):
    """Test that a job killing the runner's process each time is skipped on restart, not retried."""
    config = small_experiment(tmp_path)
    runner = tmp_path / 'crashing_job.py'
    runner.write_text(ALWAYS_CRASHING_JOB)

    code = run_until_done(['--config', str(config)], attempts=3, runner=(str(runner),))

    out = pd.read_csv(tmp_path / 'out.csv')
    report = json.loads((tmp_path / 'out.json').read_text())
    assert code == 0
    assert sorted(out['split'].unique()) == [0, 2]
    assert len(out) == 4
    # RandomSelector runs each k as its own job: both jobs of split 1 crashed.
    assert len(report['out_of_memory']) == 2
    assert all("crashed the runner's process" in f for f in report['out_of_memory'])
    assert not (tmp_path / 'out.crashed').exists()
    assert not (tmp_path / 'out.running').exists()


def test_restarts_with_the_workers_the_runner_fell_back_to(tmp_path):
    """Test that a restart passes the worker count the crashed run's JSON ended with."""
    out = tmp_path / 'out.csv'
    out.with_suffix('.json').write_text(json.dumps({'config': {'n_jobs': 2}}))
    runner = tmp_path / 'record_args.py'
    calls = tmp_path / 'calls'
    runner.write_text(f'import sys\nfrom pathlib import Path\n'
                      f'p = Path({str(calls)!r})\n'
                      f'p.write_text((p.read_text() if p.exists() else "") + " ".join(sys.argv[1:]) + "\\n")\n'
                      f'raise SystemExit(0 if p.read_text().count("\\n") > 1 else 1)\n')

    code = run_until_done(['--out', str(out), '--n-jobs', '8'], attempts=3,
                          runner=(str(runner),))

    first, second = calls.read_text().splitlines()
    assert code == 0
    assert '--n-jobs 8' in first
    assert '--n-jobs 2' in second and '--n-jobs 8' not in second


def test_gives_up_after_its_attempts(tmp_path):
    """Test that a runner failing every time is started exactly `attempts` times."""
    runner = tmp_path / 'failing_runner.py'
    starts = tmp_path / 'starts'
    runner.write_text(f'from pathlib import Path\n'
                      f'p = Path({str(starts)!r})\n'
                      f'p.write_text(p.read_text() + "x" if p.exists() else "x")\n'
                      f'raise SystemExit(1)\n')

    code = run_until_done([], attempts=2, runner=(str(runner),))

    assert code == 1
    assert starts.read_text() == 'xx'
