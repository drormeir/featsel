"""
RFECV scored by AUC against RFECV scored by accuracy, split by split, per task.

Both RFECV variants choose their own count by inner cross-validation, so
each split has one test AUC per variant. Every runner CSV of one dataset
comes from configs with the same seed and splits, so the two variants pair
by split. The test is the corrected resampled t-test of Nadeau and Bengio
(2003), as in scripts/summarize_frontier.py. Each dataset is given as
NAME=CSV[,CSV...].

Usage:
    python scripts/rfecv_scoring_tests.py \
        scanb=results/local/frontier_scanb70_cpu.csv,results/local/frontier_scanb70_rfecv_auc.csv \
        breastmnist=results/local/frontier_breastmnist70_cpu.csv,results/local/frontier_breastmnist70_rfecv_auc.csv \
        breastmnist_nocopies=results/local/frontier_breastmnist70_nocopies.csv \
        aircraft=results/aircraft70_gpu.csv \
        --out detailed_report/figures/rfecv_scoring_tests.csv
"""

import argparse

import pandas as pd
from summarize_frontier import corrected_test

BY_AUC = 'RFECV_newton_auc'
BY_ACCURACY = 'RFECV_newton'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datasets', nargs='+', help='NAME=CSV[,CSV...]')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.datasets:
        name, paths = spec.split('=')
        df = pd.concat([pd.read_csv(path) for path in paths.split(',')], ignore_index=True)
        df = df[df['selector'].isin([BY_AUC, BY_ACCURACY])]
        n_test = df['n_test'].iloc[0]
        n_train = df['n_train'].iloc[0]
        for task, group in df.groupby('task'):
            per_split = group.pivot_table(index='split', columns='selector', values='roc_auc')
            diff = (per_split[BY_AUC] - per_split[BY_ACCURACY]).dropna()
            mean, low, high, p = corrected_test(diff, n_test, n_train)
            rows.append({'dataset': name, 'task': task,
                         'auc_by_auc': per_split[BY_AUC].mean(),
                         'auc_by_accuracy': per_split[BY_ACCURACY].mean(),
                         'diff': mean, 'ci_low': low, 'ci_high': high, 'p_value': p,
                         'wins': int((diff > 0).sum()), 'splits': len(diff)})
    tests = pd.DataFrame(rows)
    tests.to_csv(args.out, index=False)
    print(tests.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
