"""
How the runner's time scales with its number of workers (task 3).

The grid of scripts/benchmark_speed.py (Lasso and RFE at k = 10, 50, 250,
5 splits, logistic regression with fixed C) on all genes and all five PAM50
subtype-vs-rest tasks: 50 jobs, enough for every worker count tried. Each
worker count runs `repeats` times. The repeats are the outer loop, so a
drift in the machine's speed spreads over every worker count instead of
falling on one.

Each run is a fresh process, so it starts from clean memory: on full
SCAN-B, worker processes left by one run can starve the next. A run counts
only if it ended with the workers it started with: the runner halves its
workers when one dies of memory exhaustion, which would mix two worker
counts in one time. A run whose process crashed counts as lost too.

Appends one row per run to --csv as soon as the run ends, with a JSON of
the settings and the machine beside it, and prints the median time, the
speed-up over one worker and the efficiency (speed-up per worker) for each
worker count.

Usage:
    python scripts/parallel_scaling.py --workers 1 2 4 6 12 --repeats 3
"""

import argparse
import json
import multiprocessing
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_speed import K_VALUES, LASSO, MODEL, N_SPLITS, RFE_PARAMS, SEED

from featsel import environment
from featsel.run import run
from featsel_datasets.scanb_gse96058.featsel_view import load_xy

ROOT = Path(__file__).resolve().parent.parent


def timed_run(data, n_jobs, folder, result):
    """Time one runner pass over the grid; write its seconds and final workers to result."""
    X, y = pd.read_pickle(data)
    out = Path(folder) / f'run_{n_jobs}.csv'
    out.unlink(missing_ok=True)
    config = {
        'output': str(out),
        'seed': SEED,
        'n_splits': N_SPLITS,
        'train_sizes': [0.5],
        'metrics': ['macro_f1'],
        'selection_repeats': 1,
        'n_jobs': n_jobs,
        'selectors': [{'name': 'LassoSelector', 'k': K_VALUES, 'params': LASSO},
                      {'name': 'RFESelector', 'k': K_VALUES, 'params': RFE_PARAMS}],
        'models': [{'name': 'logistic_regression', 'params': MODEL}],
    }
    start = time.perf_counter()
    run(X, y, config)
    seconds = time.perf_counter() - start
    ended_with = json.loads(out.with_suffix('.json').read_text())['config']['n_jobs']
    Path(result).write_text(json.dumps([seconds, ended_with]))


def isolated_run(data, n_jobs, folder):
    """timed_run in a fresh process; (NaN, 0) if that process crashed."""
    result = Path(folder) / 'result.json'
    result.unlink(missing_ok=True)
    process = multiprocessing.get_context('spawn').Process(
        target=timed_run, args=(data, n_jobs, folder, result))
    process.start()
    process.join()
    if process.exitcode != 0:
        return float('nan'), 0
    seconds, ended_with = json.loads(result.read_text())
    return seconds, ended_with


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 6, 12],
                        help='Worker counts to time; must include 1, the reference')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--n-genes', type=int,
                        help='Random genes to keep, for a quick check; default all')
    parser.add_argument('--csv', type=Path,
                        default=ROOT / 'results' / 'local' / 'parallel_scaling.csv')
    args = parser.parse_args()

    X, y = load_xy(target='pam50 subtype')
    if args.n_genes:
        genes = np.sort(np.random.default_rng(SEED).choice(X.shape[1], args.n_genes,
                                                           replace=False))
        X = X.iloc[:, genes]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.csv.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        data = Path(folder) / 'data.pkl'
        pd.to_pickle((X, y), data)
        del X, y
        for repeat in range(args.repeats):
            for n_jobs in args.workers:
                seconds, ended_with = isolated_run(data, n_jobs, folder)
                row = pd.DataFrame([{'workers': n_jobs, 'repeat': repeat, 'seconds': seconds,
                                     'ended_with_workers': ended_with}])
                row.to_csv(args.csv, mode='a', header=not args.csv.exists(), index=False)
        n_patients, n_genes = pd.read_pickle(data)[0].shape

    table = pd.read_csv(args.csv)
    environment.write_json(args.csv.with_suffix('.json'), workers=args.workers,
                           repeats=args.repeats, n_genes=n_genes, n_patients=n_patients,
                           grid={'selectors': ['LassoSelector', 'RFESelector'],
                                 'k': K_VALUES, 'n_splits': N_SPLITS, 'tasks': 5})

    valid = table[table.workers == table.ended_with_workers]
    summary = valid.groupby('workers').seconds.agg(['median', 'min', 'max', 'count'])
    summary['speed_up'] = summary.loc[1, 'median'] / summary['median']
    summary['efficiency'] = summary['speed_up'] / summary.index
    print(f'{n_patients} patients x {n_genes} genes; '
          f'{len(table) - len(valid)} runs lost workers or crashed and are left out.')
    print(summary.round(2).to_string())


if __name__ == '__main__':
    main()
