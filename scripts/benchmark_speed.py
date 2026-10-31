"""
Time the runner against plain scikit-learn on the same grid (task 3).

docs/experiment_design.md 2e. The same splits, selectors, k values and
classifier run four ways: scikit-learn's cross_validate on one core and on
every core, and the runner on one core and on every core. Every test score
must match between the two, so only the time differs.

scikit-learn refits imputation, scaling and the selector for every k. The
runner prepares each split once and ranks once per split for every k. The
time of the two one-core runs therefore separates that saving from the
saving of parallel work.

Usage:
    python scripts/benchmark_speed.py --n-genes 3000 --out docs/speed_benchmark.md
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from featsel.run import run
from featsel_datasets.scanb_gse96058.featsel_view import load_xy

SEED = 42
N_SPLITS = 5
K_VALUES = [10, 50, 250]
TASK = 'Basal'

# Fixed seeds in every estimator, so both sides fit the same models on
# every split. The runner would otherwise derive a seed per split.
LASSO = {'C': 1.0, 'random_state': SEED}
RFE_PARAMS = {'C': 0.01, 'step': 0.5, 'random_state': SEED}
MODEL = {'max_iter': 1000, 'random_state': SEED}


def sklearn_selectors(k):
    """scikit-learn's own equivalent of each runner selector, at k."""
    return {
        'LassoSelector': SelectFromModel(
            LogisticRegression(l1_ratio=1, C=LASSO['C'], solver='liblinear',
                               max_iter=1000, random_state=SEED),
            max_features=k, threshold=np.finfo(float).tiny),
        'RFESelector': RFE(
            LinearSVC(C=RFE_PARAMS['C'], max_iter=1000, random_state=SEED),
            n_features_to_select=k, step=RFE_PARAMS['step']),
    }


def run_sklearn(X, y, n_jobs):
    """Every (selector, k) as its own cross_validate. Returns scores and seconds."""
    splits = StratifiedShuffleSplit(n_splits=N_SPLITS, train_size=0.5, random_state=SEED)
    rows = []
    start = time.perf_counter()
    for k in K_VALUES:
        for name, selector in sklearn_selectors(k).items():
            pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                             ('scale', StandardScaler()),
                             ('select', selector),
                             ('model', LogisticRegression(**MODEL))])
            scores = cross_validate(pipe, X, y, cv=splits, scoring='f1_macro', n_jobs=n_jobs)
            rows += [{'selector': name, 'k': k, 'split': s, 'macro_f1': f1}
                     for s, f1 in enumerate(scores['test_score'])]
    return pd.DataFrame(rows), time.perf_counter() - start


def run_runner(X, y, n_jobs, folder):
    """The same grid through featsel.run. Returns scores and seconds."""
    config = {
        'output': str(Path(folder) / f'runner_{n_jobs}.csv'),
        'seed': SEED,
        'n_splits': N_SPLITS,
        'train_sizes': [0.5],
        'metrics': ['macro_f1'],
        'n_jobs': n_jobs,
        'selectors': [{'name': 'LassoSelector', 'k': K_VALUES, 'params': LASSO},
                      {'name': 'RFESelector', 'k': K_VALUES, 'params': RFE_PARAMS}],
        'models': [{'name': 'logistic_regression', 'params': MODEL}],
    }
    start = time.perf_counter()
    out = run(X, y, config)
    seconds = time.perf_counter() - start
    return pd.read_csv(out)[['selector', 'k', 'split', 'macro_f1']], seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-genes', type=int, default=3000,
                        help='Random genes to keep, so the grid runs in minutes')
    parser.add_argument('--out', required=True, help='Markdown file to write')
    args = parser.parse_args()

    X, y = load_xy(target='pam50 subtype')
    n_total = X.shape[1]
    genes = np.sort(np.random.default_rng(SEED).choice(n_total, args.n_genes, replace=False))
    X = X.values[:, genes]
    y = np.where(y.values == TASK, TASK, 'rest')
    cores = os.cpu_count()

    times, scores = {}, {}
    with tempfile.TemporaryDirectory() as folder:
        for n_jobs in (1, cores):
            cpus = '1 core' if n_jobs == 1 else f'{n_jobs} cores'
            scores[f'scikit-learn, {cpus}'], times[f'scikit-learn, {cpus}'] = \
                run_sklearn(X, y, n_jobs)
            scores[f'runner, {cpus}'], times[f'runner, {cpus}'] = \
                run_runner(X, y, n_jobs, folder)

    key = ['selector', 'k', 'split']
    reference = scores['scikit-learn, 1 core'].set_index(key).sort_index().macro_f1
    for label, frame in scores.items():
        other = frame.set_index(key).sort_index().macro_f1
        np.testing.assert_allclose(other.values, reference.values, err_msg=label)

    base = times['scikit-learn, 1 core']
    lines = [
        '| Run | Seconds | Speed-up over scikit-learn on 1 core |',
        '|---|---|---|',
        *[f'| {label} | {seconds:.1f} | {base / seconds:.1f} |'
          for label, seconds in times.items()],
    ]
    header = (f'SCAN-B, {TASK} vs rest, {args.n_genes} random genes of {n_total}, '
              f'{len(y)} patients. {N_SPLITS} splits at train fraction 0.5. Lasso and RFE '
              f'(step 0.5) at k = {", ".join(map(str, K_VALUES))}, logistic regression with '
              f'fixed C. {cores} cores. All {len(reference)} test macro-F1 scores are equal '
              f'on every run.\n\n')
    Path(args.out).write_text(header + '\n'.join(lines) + '\n')
    print(header + '\n'.join(lines))


if __name__ == '__main__':
    main()
