"""
Tree importance: how much the ranking depends on the forest's random seed,
at 50 to 1,000 trees.

The main runs use 200 trees, a count set by hand. For each count, two
forests with different seeds are fitted on the same training set, and the
overlap of their top-k features is measured by the Jaccard index (shared
features over all features in either set). An overlap near 1 means the seed
barely matters, so more trees would change little. Training sets only: the
test set is not used, so nothing here chooses a setting by test AUC.

Training sets of 3 random stratified splits with 70% of the samples, seed
42, as in the main runs. SCAN-B: the five subtype-vs-rest tasks, split on
the five-class label; BreastMNIST: malignant; FGVC-Aircraft: Boeing 737.

    .venv/bin/python scripts/tree_count_stability.py --out results/tree_count_stability.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from featsel import environment
from featsel.selectors import TreeImportanceSelector
from featsel_datasets.fgvc_aircraft.featsel_view import load_xy as aircraft_xy
from featsel_datasets.medmnist_breastmnist.featsel_view import load_xy as breastmnist_xy
from featsel_datasets.scanb_gse96058.featsel_view import load_xy as scanb_xy

SEED = 42
SPLITS = 3
TREES = (50, 100, 200, 500, 1000)
FOREST_SEEDS = (0, 1)


def ranking(X, y, trees, seed):
    """Feature indices by impurity decrease, largest first."""
    selector = TreeImportanceSelector(n_features=1, n_estimators=trees, random_state=seed,
                                      n_jobs=os.cpu_count())
    return selector.fit(X, y).ranking()


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def datasets():
    """(name, X, labels to split on, {task: positive mask}, top-k sizes)."""
    X, y = scanb_xy()
    yield 'SCAN-B', X.to_numpy(np.float32), y.to_numpy(), \
        {f'{c} vs rest': (y == c).to_numpy() for c in sorted(y.unique())}, (10, 50, 250, 1000)
    X, y = breastmnist_xy()
    yield 'BreastMNIST', X.to_numpy(np.float32), y.to_numpy(), \
        {'malignant': (y == 'malignant').to_numpy()}, (10, 50, 250)
    X, y = aircraft_xy()
    yield 'FGVC-Aircraft', X.to_numpy(np.float32), y.to_numpy(), \
        {'Boeing 737': (y == 'Boeing 737').to_numpy()}, (10, 50, 250)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--out', type=Path, required=True, help='Output CSV')
    args = parser.parse_args()

    rows = []
    for name, X, strata, tasks, ks in datasets():
        splits = StratifiedShuffleSplit(SPLITS, train_size=0.7, random_state=SEED)
        for split, (train, _) in enumerate(splits.split(X, strata)):
            for task, positive in tasks.items():
                for trees in TREES:
                    first, second = (ranking(X[train], positive[train], trees, s)
                                     for s in FOREST_SEEDS)
                    rows += [{'dataset': name, 'task': task, 'split': split, 'trees': trees,
                              'k': k, 'jaccard': jaccard(first[:k], second[:k])} for k in ks]
                print(name, task, split, flush=True)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    environment.write_json(args.out.with_suffix('.json'), seed=SEED, splits=SPLITS,
                           trees=TREES, forest_seeds=FOREST_SEEDS)
    print(table.pivot_table(index=['dataset', 'k'], columns='trees', values='jaccard').round(3))


if __name__ == '__main__':
    main()
