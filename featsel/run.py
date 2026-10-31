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
import importlib
import json
import resource
import shutil
import sys
import tempfile
import time
import tracemalloc
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
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


def build_tasks(y):
    """
    Expand the target into the binary classification tasks to run.

    Every task is binary (SCOPE.md 3a). A binary target is one task, 'binary'.
    A target with more classes gives one task per class, that class against
    everything else, named 'ovr_<class>'. Every task uses every sample.

    The tasks are decompositions of the same target, not independent
    datasets, and the report must say so.

    Returns a list of (task_name, y_task) pairs.
    """
    y = np.asarray(y)
    labels = sorted(pd.Series(y).unique())
    if len(labels) == 2:
        return [('binary', y)]
    return [(f'ovr_{label}', np.where(y == label, str(label), 'rest')) for label in labels]


# Selectors whose top k of feature_importances_, fit at the smallest k, is
# their selection at every larger k. Most do not use n_features in fit at
# all; RFESelector eliminates down to it, and its ranking covers every larger
# k. The runner fits them once per split and cuts the ranking at each k: the
# same genes, one fit instead of one per k. Their selection_time_s and
# selection_peak_mb are that one fit's, on every k.
RANK_ONCE = {'LassoSelector', 'MannWhitneySelector', 'RFESelector',
             'TreeImportanceSelector'}


class WorkerDiedError(RuntimeError):
    """A worker process died mid-split, most likely out of memory."""


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
    'metrics': list(METRICS),
    'n_jobs': 1,
    'max_worker_memory_gb': None,
    'tuning': {'n_splits': 3, 'train_size': 0.5, 'scoring': 'roc_auc'},
}


def load_config(path):
    """
    Read the experiment YAML. Defaults are filled in by run().

    The 'output' path, and a 'dataset' given as a YAML file, are relative to
    the config file, so a config works from any working directory.
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    base = Path(path).parent
    if 'output' in config:
        config['output'] = str(base / config['output'])
    if _is_dataset_file(config.get('dataset', '')):
        config['dataset'] = str(base / config['dataset'])
    return config


def _is_dataset_file(dataset):
    """A YAML dataset config, as opposed to a dataset module name."""
    return dataset.endswith(('.yaml', '.yml'))


def load_dataset(config):
    """
    Load X, y, the target name and the dataset name.

    'dataset' is either a module, such as
    'featsel_datasets.scanb_gse96058.featsel_view', or a CSV dataset config.
    A module defines NAME, TARGET and load_xy(target=...); its choices about
    samples and genes live in load_xy. A CSV config is read by the DataLoader,
    and unlabelled samples are dropped.
    """
    if not _is_dataset_file(config['dataset']):
        view = importlib.import_module(config['dataset'])
        target = config.get('target') or view.TARGET
        X, y = view.load_xy(target=target)
        return X, y, target, view.NAME

    loader = DataLoader(config['dataset'])
    if config.get('target'):
        loader.set_target(config['target'])

    X, y = loader.X, loader.y
    labelled = y.notna()
    if not labelled.all():
        print(f"Dropping {(~labelled).sum()} samples with no {loader.target_column} label")
        X, y = X.loc[labelled], y.loc[labelled]

    return X, y, loader.target_column, loader.config['name']


def _key(values):
    """
    An execution's key as text, as the CSV writer writes it: None is ''.

    Read back as text too, a key matches whatever else is in its column; read
    as numbers, an empty k turns the column to floats and 10 into 10.0.
    """
    return tuple('' if v is None else str(v) for v in values)


def _done_keys(path):
    """Read the keys of executions already present in an output CSV."""
    if not Path(path).exists():
        return set()
    done = pd.read_csv(path, usecols=KEY_COLUMNS, dtype=str, keep_default_na=False)
    return set(done.itertuples(index=False, name=None))


def _selection_rows(X_train, X_test, y_train, y_test, context, sel_spec, k_values,
                    config, skip, truth, failures):
    """
    Yield one row per (k, model) for one selector on one prepared task.

    X_train and X_test are already imputed and scaled on the training half, so
    nothing about the validation half has touched them.

    A selector or model that runs out of memory skips its cells and appends a
    description to failures; the rest of the job goes on.
    """
    split_seed = context['seed']
    metrics = {name: METRICS[name] for name in config['metrics']}
    sel_name = _selector_label(sel_spec)
    rank_once = (FeatureSelector.lookup(sel_spec['name']).__name__ in RANK_ONCE
                 and None not in k_values)
    ranking = None

    for k in k_values:
        keys = {_label(m): (*context['key'], sel_name, k, _label(m))
                for m in config['models']}
        if all(_key(key) in skip for key in keys.values()):
            continue

        if ranking is None:
            selector = _make_selector(sel_spec, min(k_values) if rank_once else k,
                                      split_seed)
            tracemalloc.start()
            start = time.perf_counter()
            try:
                selector.fit(X_train, y_train)
            except MemoryError:
                failures.append(f"{context['task']} split {context['split']} "
                                f"train_size {context['train_size']}: {sel_name} k={k}")
                continue
            finally:
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            selection_time = time.perf_counter() - start
            if rank_once:
                # The same argsort as FeatureSelector._top, so each cut
                # equals a fit with n_features=k.
                ranking = np.argsort(selector.feature_importances_)[::-1]

        if rank_once:
            support = np.sort(ranking[:k])
        else:
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
            if _key(keys[model_name]) in skip:
                continue

            model = _instantiate(MODELS, model_spec, split_seed)
            if 'grid' in model_spec:
                model = _tuned(model, model_spec['grid'], config['tuning'], split_seed)
            try:
                start = time.perf_counter()
                model.fit(X_train_sel, y_train)
                fit_time = time.perf_counter() - start

                start = time.perf_counter()
                y_pred = model.predict(X_test_sel)
                predict_time = time.perf_counter() - start
            except MemoryError:
                failures.append(f"{context['task']} split {context['split']} "
                                f"train_size {context['train_size']}: "
                                f"{sel_name} k={k} {model_name}")
                continue

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
                'tuned_params': json.dumps(getattr(model, 'best_params_', {})),
                'selected_indices': ' '.join(str(i) for i in support),
                **recovery,
            }
            row.update({name: fn(y_test, y_pred) for name, fn in metrics.items()})
            yield row


def _tuned(model, grid, tuning, seed):
    """
    Wrap a model so that fit chooses its parameters from grid on inner splits.

    The inner splits are stratified Monte Carlo draws from the training half
    only (docs/experiment_design.md 2b). fit then refits on the whole
    training half with the best values, and predict uses that refit model.
    """
    inner = StratifiedShuffleSplit(n_splits=tuning['n_splits'],
                                   train_size=tuning['train_size'], random_state=seed)
    return GridSearchCV(model, grid, scoring=tuning['scoring'], cv=inner)


def _job_rows(folder, y_train, y_test, context, sel_spec, k_values, config, skip, truth):
    """
    Every row of one job, and the cells that ran out of memory. The unit of
    parallel work: one selector, at the given k values, on one task of one
    prepared split.

    folder holds the split's prepared X_train.npy and X_test.npy. They are
    opened read-only and memory-mapped, so all workers share one copy.

    With config['max_worker_memory_gb'] set, the process's address space is
    capped at that size while the job runs. An allocation past the cap
    raises MemoryError, which skips the cell instead of letting the kernel
    kill a process. The cap counts reserved memory, including the mapped data
    and math-library thread buffers, so it needs headroom above real use.
    """
    limit_gb = config['max_worker_memory_gb']
    previous = resource.getrlimit(resource.RLIMIT_AS)
    if limit_gb is not None:
        resource.setrlimit(resource.RLIMIT_AS, (int(limit_gb * 1024 ** 3), previous[1]))

    rows, failures = [], []
    try:
        X_train = np.load(folder / 'X_train.npy', mmap_mode='r')
        X_test = np.load(folder / 'X_test.npy', mmap_mode='r')
        rows.extend(_selection_rows(X_train, X_test, y_train, y_test, context, sel_spec,
                                    k_values, config, skip, truth, failures))
    finally:
        # Workers are reused for later jobs and later runs, so lift the cap.
        resource.setrlimit(resource.RLIMIT_AS, previous)
    return rows, failures


def _k_groups(sel_spec):
    """
    The k values of a selector, grouped into jobs.

    A selector that ranks once per split keeps all its k values in one job,
    so it fits once. Any other selector fits once per k, so each k is its own
    job and a slow selector spreads over the workers.
    """
    k_values = sel_spec.get('k', [None])
    if FeatureSelector.lookup(sel_spec['name']).__name__ in RANK_ONCE:
        return [k_values]
    return [[k] for k in k_values]


def _prepare(X_train, X_test, preprocess, seed, folder):
    """Scale an imputed split, fit on the training half, and save it to folder."""
    scaler = PREPROCESSORS[preprocess](seed)
    if scaler is not None:
        X_train = scaler.fit(X_train).transform(X_train)
        X_test = scaler.transform(X_test)
    folder.mkdir()
    np.save(folder / 'X_train.npy', X_train)
    np.save(folder / 'X_test.npy', X_test)


def iter_rows(X, y, config, failures, skip=frozenset(), truth=None):
    """
    Yield one result row per execution, and append each cell skipped for lack
    of memory to failures.

    An execution is one (task, split, train size, preprocessing, selector, k,
    classifier) cell. Selection and preprocessing are fit on the training half
    only, inside the split, so nothing about the validation half leaks into the
    choice of features (Ambroise and McLachlan, PNAS 2002).

    Each split is stratified on the full target and shared by every task, so
    all tasks are scored on the same patients. Imputation and scaling do not
    see the labels, so they are fit once per split, in this process, and the
    result is saved to shared memory (/dev/shm where it exists). Jobs of one
    selector on one task then run on config['n_jobs'] workers, which map the
    saved split instead of copying it. A split's files are deleted once its
    last job returns; joblib prepares only a few jobs ahead, so only a couple
    of splits are on disk at once.

    Rows come back in job order, so the output does not depend on the number
    of workers.
    """
    seed = config['seed']
    X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)
    tasks = build_tasks(y_array)
    shm = Path('/dev/shm')
    shared = Path(tempfile.mkdtemp(prefix='featsel_', dir=shm if shm.is_dir() else None))
    # The folder of each dispatched job, in dispatch order, and how many of
    # each folder's jobs have not returned yet.
    job_folders = deque()
    pending = {}

    def jobs():
        for train_size in config['train_sizes']:
            splitter = StratifiedShuffleSplit(
                n_splits=config['n_splits'], train_size=train_size, random_state=seed
            )
            for split, (train_idx, test_idx) in enumerate(splitter.split(X_array, y_array)):
                imputed = None
                for preprocess in config['preprocess']:
                    # Every stochastic step in this split derives from one seed,
                    # so the split is reproducible and selectors redraw per split.
                    context = {'split': split, 'train_size': train_size,
                               'seed': seed + split, 'preprocess': preprocess}
                    cells = []
                    for task_name, y_task in tasks:
                        key = (task_name, split, train_size, preprocess)
                        for sel_spec in config['selectors']:
                            sel_name = _selector_label(sel_spec)
                            for k_values in _k_groups(sel_spec):
                                done = all(
                                    _key((*key, sel_name, k, _label(m))) in skip
                                    for k in k_values for m in config['models'])
                                if not done:
                                    cells.append((task_name, y_task, key, sel_spec, k_values))
                    if not cells:
                        continue

                    if imputed is None:
                        imputer = SimpleImputer(strategy='median').fit(X_array[train_idx])
                        imputed = (imputer.transform(X_array[train_idx]),
                                   imputer.transform(X_array[test_idx]))
                    folder = shared / f'{train_size}_{split}_{preprocess}'
                    _prepare(*imputed, preprocess, context['seed'], folder)
                    pending[folder] = len(cells)

                    for task_name, y_task, key, sel_spec, k_values in cells:
                        job_folders.append(folder)
                        yield delayed(_job_rows)(
                            folder, y_task[train_idx], y_task[test_idx],
                            {**context, 'task': task_name, 'key': key},
                            sel_spec, k_values, config, skip, truth,
                        )

    try:
        for rows, job_failures in Parallel(n_jobs=config['n_jobs'],
                                           return_as='generator')(jobs()):
            folder = job_folders.popleft()
            pending[folder] -= 1
            if not pending[folder]:
                shutil.rmtree(folder)
            failures.extend(job_failures)
            yield from rows
    finally:
        shutil.rmtree(shared, ignore_errors=True)


def print_plan(config, X, y, tasks, target, name, total):
    """Print every axis of the experiment before it starts."""
    n_features = X.shape[1]
    n_samples = X.shape[0]

    print(f"\n=== {config.get('output')} ===")
    print("Dataset      : " + (f"{name}, " if name else '')
          + f"{n_samples} samples x {n_features} features"
          + (f", target '{target}'" if target else ''))
    print(f"Seed         : {config['seed']}")

    print(f"Tasks        : {len(tasks)}")
    for task_name, y_task in tasks:
        counts = pd.Series(y_task).value_counts()
        shown = ', '.join(f"{label} {count}" for label, count in counts.items())
        print(f"  {task_name:<22} {shown}")

    print(f"Splits       : {config['n_splits']} stratified Monte Carlo draws")
    for train_size in config['train_sizes']:
        n_train = round(train_size * n_samples)
        print(f"  train_size {train_size}: {n_train} train / "
              f"{n_samples - n_train} validation")

    print(f"Preprocess   : {', '.join(config['preprocess'])}")
    print(f"Metrics      : {', '.join(config['metrics'])}")
    print(f"Workers      : {config['n_jobs']} (parallel over task x selector jobs, "
          f"one shared copy of each split)")
    if config['max_worker_memory_gb'] is not None:
        print(f"Memory cap   : {config['max_worker_memory_gb']} GB per worker")

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
        if 'grid' in spec:
            print(f"  {'':<16} tuned over {spec['grid']}")
    if any('grid' in spec for spec in config['models']):
        tuning = config['tuning']
        print(f"Tuning       : {tuning['n_splits']} inner splits at train_size "
              f"{tuning['train_size']} of each training half, best {tuning['scoring']}")

    print(f"\nTotal        : {total} executions "
          f"({len(tasks)} tasks x {config['n_splits']} splits x "
          f"{len(config['train_sizes'])} train sizes x "
          f"{len(config['preprocess'])} preprocess x "
          f"{sum(len(s.get('k', [None])) for s in config['selectors'])} selector-k x "
          f"{len(config['models'])} models)\n")


def run(X, y, config, resume=False, target=None, truth=None, name=None):
    """
    Run the experiment on data in memory, appending each finished execution to
    the output CSV.

    X is a samples-by-features array or DataFrame and y the labels. config is
    the experiment description and must name an 'output' CSV; its 'dataset'
    entry, if any, is ignored here. target and name only label the printed
    plan.

    truth, when given, holds the column indices of the truly informative
    features, as a simulation knows them. Each row then also records
    truth_recall (share of them selected) and truth_precision (share of the
    selection that is truly informative).
    """
    config = {**DEFAULTS, **config}
    config['tuning'] = {**DEFAULTS['tuning'], **config['tuning']}

    out_path = Path(config['output'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = _done_keys(out_path) if resume else set()
    if resume and skip:
        print(f"Resuming: {len(skip)} executions already in {out_path}")
    elif not resume and out_path.exists():
        out_path.unlink()

    n_k = sum(len(s.get('k', [None])) for s in config['selectors'])
    tasks = build_tasks(y)
    total = (len(tasks) * config['n_splits'] * len(config['train_sizes'])
             * len(config['preprocess']) * n_k * len(config['models']))
    print_plan(config, X, y, tasks, target, name, total)

    written, start = 0, time.perf_counter()
    writer = None
    failures = []
    has_header = out_path.exists() and out_path.stat().st_size > 0
    # The bar counts every execution in the grid, including any skipped by
    # --resume, so its total matches the number printed above.
    progress = tqdm(total=total, initial=len(skip), unit='exec',
                    smoothing=0.05, dynamic_ncols=True)
    try:
        with progress, open(out_path, 'a', newline='') as handle:
            for row in iter_rows(X, y, config, failures, skip=skip, truth=truth):
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
    except TerminatedWorkerError:
        raise WorkerDiedError(
            "Stopped: a worker process died, most likely out of memory. Compiled "
            "code such as liblinear aborts when memory is refused, instead of "
            "raising MemoryError, so the cell could not be skipped.\n"
            f"{written} executions are saved in {out_path}.\n"
            "Rerun with --resume and fewer workers (--n-jobs) or a higher "
            "--max-worker-memory-gb."
        ) from None

    print(f"Wrote {written} executions to {out_path} "
          f"in {time.perf_counter() - start:.0f}s")
    if failures:
        print(f"\nOut of memory, skipped ({len(failures)}):")
        for failure in failures:
            print(f"  {failure}")
        print("Rerun with --resume and fewer workers or a higher memory cap "
              "to fill them in.")
    return out_path


def run_config(config, resume=False):
    """Load the dataset named in the config, then run the experiment on it."""
    X, y, target, name = load_dataset(config)
    return run(X, y, config, resume=resume, target=target, name=name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to the experiment YAML')
    parser.add_argument('--out', help='Override the output CSV path')
    parser.add_argument('--resume', action='store_true',
                        help='Skip executions already present in the output CSV')
    parser.add_argument('--n-splits', type=int, help='Override the number of splits')
    parser.add_argument('--n-jobs', type=int,
                        help='Worker processes; splits run in parallel. -1 uses every core')
    parser.add_argument('--max-worker-memory-gb', type=float,
                        help='Cap each worker at this many GB; a cell past it is skipped, '
                             'not killed')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.out:
        config['output'] = args.out
    if args.n_splits:
        config['n_splits'] = args.n_splits
    if args.n_jobs:
        config['n_jobs'] = args.n_jobs
    if args.max_worker_memory_gb:
        config['max_worker_memory_gb'] = args.max_worker_memory_gb

    try:
        run_config(config, resume=args.resume)
    except WorkerDiedError as error:
        sys.exit(f"\n{error}")


if __name__ == '__main__':
    main()
