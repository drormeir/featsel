"""Tests for scripts/run_until_done.py: resuming the runner after a real crash."""

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
