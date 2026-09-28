"""
Config-driven experiment runner.

One YAML file describes the dataset, the resampling scheme, the metrics, the
feature selectors and the classifiers. The runner takes the product of those
axes and writes one CSV row per execution, where an execution is a single
(split, selector, k, classifier) cell.

Rows are appended as they finish, so a killed run can be resumed with --resume
instead of restarting. Aggregation lives in `featsel.analysis`: the CSV is
long-format, so any aggregation is a groupby afterwards.

The core, `run(X, y, config)`, takes data already in memory. `run_config`
loads the dataset named in the config first, and is what the command line uses.

Usage:
    python -m featsel.run --config configs/experiment_scanb.yaml
    python -m featsel.run --config configs/experiment_scanb.yaml --resume --n-jobs -1
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
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from .data_loader import DataLoader
from .metrics import KEY_COLUMNS, METRICS
from .selectors import FeatureSelector

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


def build_tasks(y, framings):
    """
    Expand the target into the classification tasks to run.

    'multiclass'  : the target as it stands, all classes at once.
    'one_vs_rest' : one binary task per class, that class against everything
                    else. Uses every sample.
    'one_vs_one'  : one binary task per pair of classes, using only the samples
                    belonging to those two. Isolates which subtypes are
                    genuinely hard to tell apart, which one-vs-rest hides
                    behind the majority class.

    All framings are decompositions of the same target, not independent
    datasets, and the report must say so.

    Returns a list of (task_name, y_task, mask) triples, where mask selects the
    samples the task uses and y_task is already restricted to them.
    """
    y = np.asarray(y)
    labels = sorted(pd.Series(y).unique())
    everything = np.ones(len(y), dtype=bool)
    tasks = []

    for framing in framings:
        if framing == 'multiclass':
            tasks.append(('multiclass', y, everything))

        elif framing == 'one_vs_rest':
            for label in labels:
                binary = np.where(y == label, str(label), 'rest')
                tasks.append((f'ovr_{label}', binary, everything))

        elif framing == 'one_vs_one':
            for i, first in enumerate(labels):
                for second in labels[i + 1:]:
                    mask = np.isin(y, [first, second])
                    tasks.append((f'ovo_{first}_vs_{second}', y[mask].astype(str), mask))

        else:
            raise ValueError(f"Unknown task framing '{framing}'. Available: "
                             "'multiclass', 'one_vs_rest', 'one_vs_one'")

    return tasks


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
    """Config label for a model: its name, or an explicit label."""
    return spec.get('label', spec['name'])


def _selector_label(spec):
    """
    Config label for a selector: an explicit label, or its class name.

    The class name, not the config's spelling or alias, so that results stay
    comparable whatever name a config used.
    """
    return spec.get('label') or FeatureSelector.lookup(spec['name']).__name__


def _make_selector(spec, k, seed):
    """Build a selector from its config entry, seeding it if it takes a seed."""
    cls = FeatureSelector.lookup(spec['name'])
    params = dict(spec.get('params') or {})
    if k is not None:
        params['n_features'] = k
    if 'random_state' in cls().get_params():
        params.setdefault('random_state', seed)
    return cls(**params)


DEFAULTS = {
    'seed': 42,
    'n_splits': 100,
    'train_sizes': [0.5],
    'preprocess': ['standard'],
    'task_framings': ['multiclass'],
    'metrics': list(METRICS),
    'n_jobs': 1,
}


def load_config(path):
    """
    Read the experiment YAML. Defaults are filled in by run().

    The 'dataset' and 'output' paths are relative to the config file, so a
    config works from any working directory.
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    base = Path(path).parent
    for key in ('dataset', 'output'):
        if key in config:
            config[key] = str(base / config[key])
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


def _selection_rows(X_train, X_test, y_train, y_test, context, config,
                    metrics, skip, truth):
    """
    Yield one row per (selector, k, model) for a single prepared split.

    X_train and X_test are already imputed and scaled on the training half, so
    nothing about the validation half has touched them.
    """
    split_seed = context['seed']

    for sel_spec in config['selectors']:
        sel_name = _selector_label(sel_spec)

        for k in sel_spec.get('k', [None]):
            keys = {_label(m): (*context['key'], sel_name, k, _label(m))
                    for m in config['models']}
            if all(tuple(str(v) for v in key) in skip for key in keys.values()):
                continue

            selector = _make_selector(sel_spec, k, split_seed)

            tracemalloc.start()
            start = time.perf_counter()
            selector.fit(X_train, y_train)
            selection_time = time.perf_counter() - start
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            support = selector.get_support(indices=True)
            X_train_sel, X_test_sel = X_train[:, support], X_test[:, support]

            recovery = {}
            if truth is not None:
                hits = len(np.intersect1d(support, truth))
                recovery = {
                    'truth_recall': hits / len(truth),
                    'truth_precision': hits / len(support) if len(support) else 0.0,
                }

            for model_spec in config['models']:
                model_name = _label(model_spec)
                if tuple(str(v) for v in keys[model_name]) in skip:
                    continue

                model = _instantiate(MODELS, model_spec, split_seed)
                start = time.perf_counter()
                model.fit(X_train_sel, y_train)
                fit_time = time.perf_counter() - start

                start = time.perf_counter()
                y_pred = model.predict(X_test_sel)
                predict_time = time.perf_counter() - start

                row = {
                    'task': context['task'],
                    'split': context['split'],
                    'train_size': context['train_size'],
                    'preprocess': context['preprocess'],
                    'selector': sel_name,
                    'k': k,
                    'model': model_name,
                    'seed': split_seed,
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'n_selected': len(support),
                    'selection_time_s': selection_time,
                    'selection_peak_mb': peak_bytes / 1024 ** 2,
                    'fit_time_s': fit_time,
                    'predict_time_s': predict_time,
                    'selector_params': json.dumps(sel_spec.get('params') or {}),
                    'model_params': json.dumps(model_spec.get('params') or {}),
                    'selected_indices': ' '.join(str(i) for i in support),
                    **recovery,
                }
                row.update({name: fn(y_test, y_pred) for name, fn in metrics.items()})
                yield row


def _split_rows(X_task, y_task, train_idx, test_idx, context, config, skip, truth):
    """
    Every row of one split. The unit of parallel work.

    A split is independent of every other split and carries its own seed, so
    the rows are identical whether splits run in sequence or in parallel.
    """
    metrics = {name: METRICS[name] for name in config['metrics']}
    y_train, y_test = y_task[train_idx], y_task[test_idx]

    imputer = SimpleImputer(strategy='median').fit(X_task[train_idx])
    X_train_imp = imputer.transform(X_task[train_idx])
    X_test_imp = imputer.transform(X_task[test_idx])

    rows = []
    for preprocess in config['preprocess']:
        scaler = PREPROCESSORS[preprocess](context['seed'])
        if scaler is None:
            X_train, X_test = X_train_imp, X_test_imp
        else:
            scaler.fit(X_train_imp)
            X_train = scaler.transform(X_train_imp)
            X_test = scaler.transform(X_test_imp)

        prepared = {**context, 'preprocess': preprocess,
                    'key': (context['task'], context['split'], context['train_size'], preprocess)}
        rows.extend(_selection_rows(X_train, X_test, y_train, y_test,
                                    prepared, config, metrics, skip, truth))
    return rows


def iter_rows(X, y, config, skip=frozenset(), truth=None):
    """
    Yield one result row per execution.

    An execution is one (task, split, train size, preprocessing, selector, k,
    classifier) cell. Selection and preprocessing are fit on the training half
    only, inside the split, so nothing about the validation half leaks into the
    choice of features (Ambroise and McLachlan, PNAS 2002).

    Splits run on config['n_jobs'] worker processes. Rows come back in split
    order, so the output does not depend on the number of workers.
    """
    seed = config['seed']
    X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)

    def jobs():
        for task_name, y_task, mask in build_tasks(y_array, config['task_framings']):
            # One-vs-one tasks use only the samples of the two classes involved.
            X_task = X_array[mask]

            for train_size in config['train_sizes']:
                splitter = StratifiedShuffleSplit(
                    n_splits=config['n_splits'], train_size=train_size, random_state=seed
                )

                for split, (train_idx, test_idx) in enumerate(splitter.split(X_task, y_task)):
                    # Every stochastic step in this split derives from one seed,
                    # so the split is reproducible and selectors redraw per split.
                    context = {'task': task_name, 'split': split,
                               'train_size': train_size, 'seed': seed + split}
                    yield delayed(_split_rows)(X_task, y_task, train_idx, test_idx,
                                               context, config, skip, truth)

    for rows in Parallel(n_jobs=config['n_jobs'], return_as='generator')(jobs()):
        yield from rows


def print_plan(config, X, y, tasks, target, total):
    """Print every axis of the experiment before it starts."""
    n_features = X.shape[1]
    n_samples = X.shape[0]

    print(f"\n=== {config.get('output')} ===")
    print(f"Dataset      : {n_samples} samples x {n_features} features"
          + (f", target '{target}'" if target else ''))
    print(f"Seed         : {config['seed']}")

    print(f"Tasks        : {len(tasks)}")
    for name, y_task, mask in tasks:
        counts = pd.Series(y_task).value_counts()
        shown = ', '.join(f"{label} {count}" for label, count in counts.items())
        used = '' if mask.all() else f", {mask.sum()} of {n_samples} samples"
        print(f"  {name:<22} {len(counts)} classes ({shown}{used})")

    print(f"Splits       : {config['n_splits']} stratified Monte Carlo draws")
    for train_size in config['train_sizes']:
        # One-vs-one tasks use fewer samples, so report the range across tasks.
        totals = sorted({int(m.sum()) for _, _, m in tasks})
        splits = [f"{round(train_size * n)} train / "
                  f"{n - round(train_size * n)} validation" for n in totals]
        if len(splits) == 1:
            print(f"  train_size {train_size}: {splits[0]}")
        else:
            print(f"  train_size {train_size}: {splits[0]} .. {splits[-1]} "
                  f"depending on the task")

    print(f"Preprocess   : {', '.join(config['preprocess'])}")
    print(f"Metrics      : {', '.join(config['metrics'])}")
    print(f"Workers      : {config['n_jobs']} (parallel over splits)")

    print("Selectors    :")
    for spec in config['selectors']:
        k_values = spec.get('k', [None])
        if k_values == [None]:
            shown = "method chooses its own count"
        else:
            shown = ', '.join(f"{k} ({100 * k / n_features:.2f}%)" for k in k_values)
        params = spec.get('params') or {}
        print(f"  {_selector_label(spec):<16} k = {shown}")
        if params:
            print(f"  {'':<16} params {params}")

    print("Models       :")
    for spec in config['models']:
        print(f"  {_label(spec):<16} {spec.get('params') or {}}")

    print(f"\nTotal        : {total} executions "
          f"({len(tasks)} tasks x {config['n_splits']} splits x "
          f"{len(config['train_sizes'])} train sizes x "
          f"{len(config['preprocess'])} preprocess x "
          f"{sum(len(s.get('k', [None])) for s in config['selectors'])} selector-k x "
          f"{len(config['models'])} models)\n")


def run(X, y, config, resume=False, target=None, truth=None):
    """
    Run the experiment on data in memory, appending each finished execution to
    the output CSV.

    X is a samples-by-features array or DataFrame and y the labels. config is
    the experiment description and must name an 'output' CSV; its 'dataset'
    entry, if any, is ignored here. target only labels the printed plan.

    truth, when given, holds the column indices of the truly informative
    features, as a simulation knows them. Each row then also records
    truth_recall (share of them selected) and truth_precision (share of the
    selection that is truly informative).
    """
    config = {**DEFAULTS, **config}

    out_path = Path(config['output'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = _done_keys(out_path) if resume else set()
    if resume and skip:
        print(f"Resuming: {len(skip)} executions already in {out_path}")
    elif not resume and out_path.exists():
        out_path.unlink()

    n_k = sum(len(s.get('k', [None])) for s in config['selectors'])
    tasks = build_tasks(np.asarray(y), config['task_framings'])
    total = (len(tasks) * config['n_splits'] * len(config['train_sizes'])
             * len(config['preprocess']) * n_k * len(config['models']))
    print_plan(config, X, y, tasks, target, total)

    written, start = 0, time.perf_counter()
    writer = None
    has_header = out_path.exists() and out_path.stat().st_size > 0
    # The bar counts every execution in the grid, including any skipped by
    # --resume, so its total matches the number printed above.
    progress = tqdm(total=total, initial=len(skip), unit='exec',
                    smoothing=0.05, dynamic_ncols=True)
    with progress, open(out_path, 'a', newline='') as handle:
        for row in iter_rows(X, y, config, skip=skip, truth=truth):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not has_header:
                    writer.writeheader()

            writer.writerow(row)
            handle.flush()  # a killed run keeps everything already finished
            written += 1
            progress.update(1)
            progress.set_postfix_str(
                f"{row['task']} {row['selector']} k={row['k']} {row['model']}",
                refresh=False,
            )

    print(f"Wrote {written} executions to {out_path} "
          f"in {time.perf_counter() - start:.0f}s")
    return out_path


def run_config(config, resume=False):
    """Load the dataset named in the config, then run the experiment on it."""
    X, y, target = load_dataset(config)
    return run(X, y, config, resume=resume, target=target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to the experiment YAML')
    parser.add_argument('--out', help='Override the output CSV path')
    parser.add_argument('--resume', action='store_true',
                        help='Skip executions already present in the output CSV')
    parser.add_argument('--n-splits', type=int, help='Override the number of splits')
    parser.add_argument('--n-jobs', type=int,
                        help='Worker processes; splits run in parallel. -1 uses every core')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.out:
        config['output'] = args.out
    if args.n_splits:
        config['n_splits'] = args.n_splits
    if args.n_jobs:
        config['n_jobs'] = args.n_jobs

    run_config(config, resume=args.resume)


if __name__ == '__main__':
    main()
