"""
Does the CorrelationSelector fix of commit 9d9d904 change the frontier selections?

The old rule dropped the weaker gene of every redundant pair, even when the
stronger one was itself dropped. The new rule walks the genes from the
largest target correlation down and drops a gene only when it is redundant
with a gene already kept. The frontier runs used the old rule.

For each run, this rebuilds every split as featsel.run does (the runner's
own dataset loader, splitter, imputer, scaler and selector factory), fits the
old selector (loaded from git) and the new one once per task and split, and
cuts each ranking at every k. It checks that the old fit gives the saved
`selected_indices`, then compares the new selection with the saved one. No
model is fit and nothing is timed.

The column `chosen` marks the k that validation chose with the old ranking
(featsel.analysis.chosen_k); the new ranking may move that choice, which
this does not recompute.

Usage:
    .venv/bin/python scripts/correlation_fix_check.py \
        --out detailed_report/figures/correlation_fix_check.csv
"""

import argparse
import importlib.util
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from featsel.analysis import chosen_k
from featsel.run import (
    DEFAULTS,
    PREPROCESSORS,
    _make_selector,
    _selected,
    _selector_label,
    build_tasks,
    load_config,
    load_dataset,
)

ROOT = Path(__file__).resolve().parent.parent

# (dataset label, config, results CSV, dataset_params the CSV was made with)
RUNS = [
    ('SCAN-B', 'configs/frontier_scanb70_gpu.yaml', 'results/local/frontier_scanb70_gpu.csv',
     None),
    ('BreastMNIST', 'configs/frontier_breastmnist70_nocopies.yaml',
     'results/local/frontier_breastmnist70_nocopies.csv', None),
    # The saved CSV has all 1,200 photos: commit e870670 set drop_copies in
    # the config after the run.
    ('FGVC-Aircraft', 'configs/aircraft70_gpu.yaml', 'results/aircraft70_gpu.csv',
     {'drop_copies': False}),
]

SELECTOR = 'CorrelationSelector'


def old_selector_class():
    """CorrelationSelector as it was before commit 9d9d904, unregistered."""
    source = subprocess.run(
        ['git', 'show', '9d9d904~1:featsel/selectors/correlation/__init__.py'],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout
    source = source.replace('class CorrelationSelector(FeatureSelector):',
                            'class CorrelationSelector(FeatureSelector, register=False):')
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'old.py'
        path.write_text(source)
        # Named inside the package, so its relative imports resolve.
        spec = importlib.util.spec_from_file_location('featsel.selectors.correlation.old', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.CorrelationSelector


def saved_selections(path):
    """The correlation rows of a runner CSV, one per (task, split, k), and the chosen k."""
    rows = pd.read_csv(path)
    rows = rows[rows['selector'] == SELECTOR]
    chosen = set(chosen_k(rows)[['task', 'split', 'k']].itertuples(index=False, name=None))
    rows = rows.drop_duplicates(['task', 'split', 'k'])
    return rows, chosen


def check_run(name, config_path, csv_path, dataset_params, old_cls):
    """One output row per (task, split, k) of one frontier run."""
    config = {**DEFAULTS, **load_config(ROOT / config_path)}
    if dataset_params is not None:
        config['dataset_params'] = dataset_params
    (spec,) = [s for s in config['selectors'] if _selector_label(s) == SELECTOR]
    k_values = spec['k']
    saved, chosen = saved_selections(ROOT / csv_path)

    X, y, _, _ = load_dataset(config)
    n_saved = saved['n_train'].iloc[0] + saved['n_test'].iloc[0]
    if len(y) != n_saved:
        raise ValueError(f"{name}: {len(y)} samples, but the saved run had {n_saved}")
    # As featsel.run.iter_rows does.
    X = np.asarray(X, dtype=np.float32)
    y = y.values if isinstance(y, pd.Series) else np.asarray(y)
    tasks = build_tasks(y)
    seed = config['seed']
    (train_size,) = config['train_sizes']
    (preprocess,) = config['preprocess']
    splitter = StratifiedShuffleSplit(n_splits=config['n_splits'], train_size=train_size,
                                      random_state=seed)

    out = []
    for split, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        split_seed = seed + split
        imputer = SimpleImputer(strategy='median').fit(X[train_idx])
        X_train = imputer.transform(X[train_idx])
        scaler = PREPROCESSORS[preprocess](split_seed)
        if scaler is not None:
            X_train = scaler.fit(X_train).transform(X_train)

        for task, y_task in tasks:
            y_train = y_task[train_idx]
            start = time.perf_counter()
            new = _selected(_make_selector(spec, min(k_values), split_seed)
                            .fit(X_train, y_train), ranked=True)
            old = _selected(old_cls(n_features=min(k_values), **spec.get('params', {}))
                            .fit(X_train, y_train), ranked=True)
            rows = saved[(saved['task'] == task) & (saved['split'] == split)]
            first = len(out)
            for k in k_values:
                (indices,) = rows.loc[rows['k'] == k, 'selected_indices']
                saved_set = {int(i) for i in str(indices).split()}
                new_set = set(new[:k].tolist())
                shared = len(saved_set & new_set)
                out.append({
                    'dataset': name, 'task': task, 'split': split, 'k': k,
                    'chosen': (task, split, k) in chosen,
                    'reproduced': set(old[:k].tolist()) == saved_set,
                    'old_count': len(saved_set), 'new_count': len(new_set),
                    'shared': shared,
                    'identical': shared == len(saved_set) == len(new_set),
                })
            cells = out[first:]
            print(f"{name} {task} split {split}: "
                  f"reproduced {sum(c['reproduced'] for c in cells)}/{len(cells)}, "
                  f"identical {sum(c['identical'] for c in cells)}/{len(cells)}, "
                  f"{time.perf_counter() - start:.1f}s", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    old_cls = old_selector_class()
    table = pd.DataFrame([row for run in RUNS for row in check_run(*run, old_cls)])
    table.to_csv(args.out, index=False)

    for name, group in table.groupby('dataset', sort=False):
        at_chosen = group[group['chosen']]
        differ = at_chosen[~at_chosen['identical']]
        print(f"\n{name}: reproduced {group['reproduced'].sum()}/{len(group)}, "
              f"identical {group['identical'].sum()}/{len(group)} (task, split, k); "
              f"at the chosen k identical {at_chosen['identical'].sum()}/{len(at_chosen)}")
        if len(differ):
            print(differ.assign(differ=differ['old_count'] - differ['shared'])
                  [['task', 'split', 'k', 'old_count', 'new_count', 'shared', 'differ']]
                  .to_string(index=False))


if __name__ == '__main__':
    main()
