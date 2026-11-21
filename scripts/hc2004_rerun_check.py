"""
Does HC2004 centred at i/(N+1) select what the saved runs, centred at i/N, did?

The HC2004 rows of the reports' runs come from the version that centred the
i-th sorted p-value at i/N. Commit 5facf3e, made after those runs, centres
it at i/(N+1). HC2004 alone was rerun on the same splits, one config per saved run
(`results/local/hc2004_rerun/*_hc.yaml`: the saved run's config with only
HC2004 left). This compares each task and split with the saved row: the
selected features and the test AUC.

Usage:
    .venv/bin/python scripts/hc2004_rerun_check.py \
        --out detailed_report/figures/hc2004_rerun_check.csv
"""

import argparse
from pathlib import Path

import pandas as pd

RERUN = Path('results/local/hc2004_rerun')
SAVED = {
    'frontier_scanb70_gpu': 'results/local/frontier_scanb70_gpu.csv',
    'frontier_breastmnist70_nocopies': 'results/local/frontier_breastmnist70_nocopies.csv',
    'aircraft70_gpu': 'results/aircraft70_gpu.csv',
    'rare_signal70': 'results/rare_signal70.csv',
}
KEY = ['task', 'split']


def compare(saved_path, rerun_path):
    saved = pd.read_csv(saved_path, keep_default_na=False, dtype=str)
    old = saved[saved.selector == 'HC2004'].set_index(KEY).sort_index()
    new = pd.read_csv(rerun_path, keep_default_na=False, dtype=str).set_index(KEY).sort_index()
    if not old.index.equals(new.index) or not (old.n_train == new.n_train).all():
        raise ValueError(f'{rerun_path}: not the cells of {saved_path}')
    return {'cells': len(old),
            'same_features': int((old.selected_indices == new.selected_indices).sum()),
            'same_roc_auc': int((old.roc_auc.astype(float) == new.roc_auc.astype(float)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [{'run': run, **compare(saved, RERUN / f'{run}_hc.csv')}
            for run, saved in SAVED.items()]
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
