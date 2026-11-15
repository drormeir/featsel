"""
Summarize the side scores and the memory of frontier runs, from saved CSVs.

The runner writes macro-F1, balanced accuracy and the peak memory of each
selection (selection_peak_mb), but summarize_frontier.py reports only ROC AUC
and time. This script reads the same CSVs and runs nothing.

At the k each method chose on validation sets (by validation ROC AUC, as in
summarize_frontier.py), one row per task and method: mean macro-F1 and its
quartiles, mean balanced accuracy, and the median and largest peak memory of
the selection. Random selection at every feature is the all-features row.
Sequential selection does not choose its k, so it has one row per k.

Usage:
    python scripts/summarize_side_metrics.py --n-features 28541 \
        --out final_report/figures/summary_scanb70_side results/local/frontier_scanb70_*.csv

Writes <out>.csv.
"""

import argparse

import pandas as pd

from featsel.analysis import chosen_k


def side_table(rows):
    """One row per task and method: side scores and peak selection memory."""
    table = rows.groupby(['task', 'selector']).agg(
        macro_f1=('macro_f1', 'mean'),
        macro_f1_q1=('macro_f1', lambda s: s.quantile(0.25)),
        macro_f1_q3=('macro_f1', lambda s: s.quantile(0.75)),
        balanced_accuracy=('balanced_accuracy', 'mean'),
        roc_auc=('roc_auc', 'mean'),
        peak_mb_median=('selection_peak_mb', 'median'),
        peak_mb_max=('selection_peak_mb', 'max'),
        splits=('split', 'nunique'),
    )
    return table.reset_index().sort_values(['task', 'roc_auc'], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='+')
    parser.add_argument('--n-features', type=int, required=True,
                        help='Features in the data; random selection at this k is every feature')
    parser.add_argument('--out', required=True, help='Path of the CSV to write, without .csv')
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    sequential = df.selector.str.startswith('Sequential')
    random = df.selector == 'RandomSelector'
    everything = df[random & (df.k == args.n_features)].assign(
        selector=f'all {args.n_features} features')
    sequential_rows = df[sequential].assign(
        selector=lambda d: d.selector + ', k = ' + d.k.astype(int).astype(str))
    rows = pd.concat([chosen_k(df[~sequential & ~random]), everything, sequential_rows])

    table = side_table(rows)
    table.to_csv(f'{args.out}.csv', index=False)
    print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
