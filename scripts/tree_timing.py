"""
Tree importance on SCAN-B: why the runs' time per fit exceeds the speed series.

The speed series (detailed report, section 6.2) fit on 1,488 patients, the
training set of a 50% split, alone on the machine. The SCAN-B runs fit on
2,083 patients, a 70% split, with 4 threads per fit and two workers at
once. This script times the same forest (200 trees) on Basal against the
rest, on the 70% training set of the runs' seed: on 1 and 4 threads alone,
and on 4 threads with two fits at once, as in the runs.

    .venv/bin/python scripts/tree_timing.py --out results/tree_timing.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from featsel import environment
from featsel.selectors import TreeImportanceSelector
from featsel_datasets.scanb_gse96058.featsel_view import load_xy

SEED = 42
TREES = 200
FITS = 4


def fit_time(n_jobs, X, y):
    """Seconds for one fit of the selector."""
    start = time.perf_counter()
    TreeImportanceSelector(n_features=10, n_estimators=TREES, n_jobs=n_jobs).fit(X, y)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--out', type=Path, required=True, help='Output CSV')
    args = parser.parse_args()

    X, y = load_xy()
    y = (y == 'Basal').to_numpy()
    split = StratifiedShuffleSplit(1, train_size=0.7, random_state=SEED)
    train, _ = next(split.split(X, y))
    X = StandardScaler().fit_transform(X.to_numpy()[train])
    y = y[train]

    rows = []
    for threads in (1, 4):
        rows += [('alone', threads, fit_time(threads, X, y)) for _ in range(FITS)]
    together = Parallel(n_jobs=2)(delayed(fit_time)(4, X, y) for _ in range(2 * FITS))
    rows += [('two at once', 4, t) for t in together]

    table = pd.DataFrame(rows, columns=['setting', 'threads', 'seconds'])
    table.insert(0, 'patients', X.shape[0])
    table.insert(1, 'genes', X.shape[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    environment.write_json(args.out.with_suffix('.json'), seed=SEED, trees=TREES, fits=FITS)
    print(table.groupby(['setting', 'threads']).seconds.describe())


if __name__ == '__main__':
    main()
