"""
Config-driven experiment runner.

One YAML file describes the dataset, the resampling scheme, the metrics, the
feature selectors and the classifiers. The runner takes the product of those
axes and writes one CSV row per execution, where an execution is a single
(split, selector, k, classifier) cell.

Rows are appended as they finish, so a killed run can be resumed with --resume
instead of restarting. Aggregation (median, Q1, Q3) is deliberately left out:
the CSV is long-format, so any aggregation is a groupby afterwards.

Usage:
    python -m featsel.run --config configs/experiment_scanb.yaml
    python -m featsel.run --config configs/experiment_scanb.yaml --resume
"""

import argparse
import csv
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .data_loader import DataLoader
from .feature_selector import FeatureSelector


def _g_mean(y_true, y_pred):
    """
    Geometric mean of per-class recalls (Kubat and Matwin, 1997).

    Stricter than macro-F1 on imbalanced targets: it collapses to zero if any
    single class is never predicted correctly.
    """
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    if np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


# name -> callable(y_true, y_pred) -> float. Add a metric with one line.
METRICS = {
    'accuracy': accuracy_score,
    'macro_f1': lambda t, p: f1_score(t, p, average='macro', zero_division=0),
    'balanced_accuracy': balanced_accuracy_score,
    'mcc': matthews_corrcoef,
    'g_mean': _g_mean,
}

# name -> sklearn estimator class. Parameters come from the config, not here,
# so adding a classifier is one line plus a config entry.
MODELS = {
    'logistic_regression': LogisticRegression,
    'linear_svm': LinearSVC,
    'random_forest': RandomForestClassifier,
    'knn': KNeighborsClassifier,
    'lda': LinearDiscriminantAnalysis,
}

try:  # optional dependency
    from xgboost import XGBClassifier

    MODELS['xgboost'] = XGBClassifier
except ImportError:  # pragma: no cover - depends on the environment
    pass

PREPROCESSORS = {
    'none': lambda seed: None,
    'standard': lambda seed: StandardScaler(),
    'quantile_normal': lambda seed: QuantileTransformer(
        output_distribution='normal', subsample=100_000, random_state=seed
    ),
}

# Columns that identify an execution. Used to skip finished work on --resume.
KEY_COLUMNS = ['split', 'train_size', 'preprocess', 'selector', 'k', 'model']


def _instantiate(registry, spec, seed):
    """Build an estimator from a {name, params} config entry, seeding if it takes one."""
    name = spec['name']
    if name not in registry:
        raise ValueError(f"Unknown entry '{name}'. Available: {', '.join(registry)}")

    params = dict(spec.get('params') or {})
    cls = registry[name]
    if 'random_state' in cls().get_params() and 'random_state' not in params:
        params['random_state'] = seed
    return cls(**params)


def _label(spec):
    """Config label for a selector or model: its name, or an explicit label."""
    return spec.get('label', spec['name'])


def load_config(path):
    """Read the experiment YAML and fill in defaults."""
    with open(path) as f:
        config = yaml.safe_load(f)

    config.setdefault('seed', 42)
    config.setdefault('n_splits', 100)
    config.setdefault('train_sizes', [0.5])
    config.setdefault('preprocess', ['standard'])
    config.setdefault('metrics', list(METRICS))
    config.setdefault('output', 'results/experiment.csv')
    return config


def load_dataset(config):
    """Load X and y through the DataLoader, dropping unlabelled samples."""
    loader = DataLoader(config['dataset'])
    if config.get('target'):
        loader.set_target(config['target'])

    X, y = loader.X, loader.y
    labelled = y.notna()
    if not labelled.all():
        print(f"Dropping {(~labelled).sum()} samples with no {loader.target_column} label")
        X, y = X.loc[labelled], y.loc[labelled]

    return X, y, loader.target_column


def _done_keys(path):
    """Read the keys of executions already present in an output CSV."""
    if not Path(path).exists():
        return set()
    done = pd.read_csv(path, usecols=KEY_COLUMNS)
    return {tuple(str(v) for v in row) for row in done.itertuples(index=False)}


def iter_rows(X, y, config, skip=frozenset()):
    """
    Yield one result row per (split, preprocess, selector, k, model) execution.

    Selection and preprocessing are fit on the training half only, inside the
    split, so nothing about the validation half leaks into the choice of
    features (Ambroise and McLachlan, PNAS 2002).
    """
    seed = config['seed']
    metrics = {name: METRICS[name] for name in config['metrics']}

    X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)

    for train_size in config['train_sizes']:
        splitter = StratifiedShuffleSplit(
            n_splits=config['n_splits'], train_size=train_size, random_state=seed
        )

        for split, (train_idx, test_idx) in enumerate(splitter.split(X_array, y_array)):
            # Every stochastic step in this split derives from one seed, so the
            # split is reproducible on its own and selectors redraw per split.
            split_seed = seed + split

            X_train_raw, X_test_raw = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            imputer = SimpleImputer(strategy='median').fit(X_train_raw)
            X_train_imp = imputer.transform(X_train_raw)
            X_test_imp = imputer.transform(X_test_raw)

            for preprocess in config['preprocess']:
                scaler = PREPROCESSORS[preprocess](split_seed)
                if scaler is None:
                    X_train, X_test = X_train_imp, X_test_imp
                else:
                    scaler.fit(X_train_imp)
                    X_train = scaler.transform(X_train_imp)
                    X_test = scaler.transform(X_test_imp)

                for sel_spec in config['selectors']:
                    sel_name = _label(sel_spec)
                    for k in sel_spec.get('k', [None]):
                        params = dict(sel_spec.get('params') or {})
                        params.setdefault('random_state', split_seed)

                        keys = [(split, train_size, preprocess, sel_name, k, _label(m))
                                for m in config['models']]
                        if all(tuple(str(v) for v in key) in skip for key in keys):
                            continue

                        selector = FeatureSelector(
                            method=sel_spec['name'], n_features=k, **params
                        )

                        tracemalloc.start()
                        start = time.perf_counter()
                        selector.fit(X_train, y_train)
                        selection_time = time.perf_counter() - start
                        _, peak_bytes = tracemalloc.get_traced_memory()
                        tracemalloc.stop()

                        support = selector.get_support(indices=True)
                        X_train_sel, X_test_sel = X_train[:, support], X_test[:, support]

                        for model_spec, key in zip(config['models'], keys):
                            if tuple(str(v) for v in key) in skip:
                                continue

                            model = _instantiate(MODELS, model_spec, split_seed)
                            start = time.perf_counter()
                            model.fit(X_train_sel, y_train)
                            fit_time = time.perf_counter() - start

                            start = time.perf_counter()
                            y_pred = model.predict(X_test_sel)
                            predict_time = time.perf_counter() - start

                            row = {
                                'split': split,
                                'train_size': train_size,
                                'preprocess': preprocess,
                                'selector': sel_name,
                                'k': k,
                                'model': _label(model_spec),
                                'seed': split_seed,
                                'n_train': len(train_idx),
                                'n_test': len(test_idx),
                                'n_selected': len(support),
                                'selection_time_s': selection_time,
                                'selection_peak_mb': peak_bytes / 1024 ** 2,
                                'fit_time_s': fit_time,
                                'predict_time_s': predict_time,
                                'selector_params': json.dumps(sel_spec.get('params') or {}),
                                'model_params': json.dumps(model_spec.get('params') or {}),
                                'selected_indices': ' '.join(str(i) for i in support),
                            }
                            row.update({name: fn(y_test, y_pred)
                                        for name, fn in metrics.items()})
                            yield row


def run(config, resume=False):
    """Run the experiment, appending each finished execution to the output CSV."""
    X, y, target = load_dataset(config)

    out_path = Path(config['output'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = _done_keys(out_path) if resume else set()
    if resume and skip:
        print(f"Resuming: {len(skip)} executions already in {out_path}")
    elif not resume and out_path.exists():
        out_path.unlink()

    n_k = sum(len(s.get('k', [None])) for s in config['selectors'])
    total = (config['n_splits'] * len(config['train_sizes']) * len(config['preprocess'])
             * n_k * len(config['models']))
    print(f"Target '{target}': {total} executions "
          f"({config['n_splits']} splits x {len(config['train_sizes'])} train sizes x "
          f"{len(config['preprocess'])} preprocess x {n_k} selector-k x "
          f"{len(config['models'])} models)")

    written, start = 0, time.perf_counter()
    writer, handle = None, None
    try:
        for row in iter_rows(X, y, config, skip=skip):
            if writer is None:
                exists = out_path.exists() and out_path.stat().st_size > 0
                handle = open(out_path, 'a', newline='')
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not exists:
                    writer.writeheader()

            writer.writerow(row)
            handle.flush()  # a killed run keeps everything already finished
            written += 1
            if written % 25 == 0:
                rate = written / (time.perf_counter() - start)
                print(f"  {written + len(skip)}/{total} executions "
                      f"({rate:.1f}/s)", flush=True)
    finally:
        if handle is not None:
            handle.close()

    print(f"Wrote {written} executions to {out_path} "
          f"in {time.perf_counter() - start:.0f}s")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to the experiment YAML')
    parser.add_argument('--out', help='Override the output CSV path')
    parser.add_argument('--resume', action='store_true',
                        help='Skip executions already present in the output CSV')
    parser.add_argument('--n-splits', type=int, help='Override the number of splits')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.out:
        config['output'] = args.out
    if args.n_splits:
        config['n_splits'] = args.n_splits

    run(config, resume=args.resume)


if __name__ == '__main__':
    main()
