"""
Summarize frontier runs at the k each method chose on validation sets.

For each task, one row per method: the mean test ROC AUC over the splits with
its quartiles, the number of features chosen, the median selection, search
and validation times, and Nogueira's stability over the splits. Then each
method against the random control, and against every feature when the run
has it, by the corrected resampled t-test of Nadeau and Bengio (2003): the
variance of the mean difference is (1/J + n_test/n_train) times the sample
variance, for J splits, with J - 1 degrees of freedom.

Sequential selection does not choose its k (it takes minutes per fit), so it
is left out of the chosen-k tables; its AUC at each k is in the curve table.

Usage:
    python scripts/summarize_frontier.py --n-features 2048 \
        --out final_report/figures/summary_breastmnist70 results/local/frontier_breastmnist70_*.csv

Writes <out>_chosen.csv, <out>_tests.csv and <out>_curve.csv.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from featsel.analysis import chosen_k, nogueira_stability


def corrected_test(diff, n_test, n_train):
    """Mean difference, its 95% interval and p-value under the corrected t-test."""
    j = len(diff)
    var = diff.var(ddof=1) * (1 / j + n_test / n_train)
    if var == 0:
        return diff.mean(), diff.mean(), diff.mean(), 1.0
    half = stats.t.ppf(0.975, j - 1) * np.sqrt(var)
    p = 2 * stats.t.sf(abs(diff.mean() / np.sqrt(var)), j - 1)
    return diff.mean(), diff.mean() - half, diff.mean() + half, p


def chosen_table(chosen, n_features):
    """One row per task and method at the chosen k."""
    stability = nogueira_stability(chosen.assign(k=0), n_features)
    stability = stability.set_index(['task', 'selector'])['stability'] if len(stability) else None
    table = chosen.groupby(['task', 'selector']).agg(
        roc_auc=('roc_auc', 'mean'),
        roc_auc_q1=('roc_auc', lambda s: s.quantile(0.25)),
        roc_auc_q3=('roc_auc', lambda s: s.quantile(0.75)),
        n_selected_median=('n_selected', 'median'),
        n_selected_min=('n_selected', 'min'),
        n_selected_max=('n_selected', 'max'),
        selection_time_s=('selection_time_s', 'median'),
        search_time_s=('search_time_s', 'median'),
        validation_time_s=('validation_time_s', 'median'),
        splits=('split', 'nunique'),
    )
    if stability is not None:
        table['stability'] = stability
    return table.reset_index().sort_values(['task', 'roc_auc'], ascending=[True, False])


def tests_table(chosen, everything, n_features):
    """Each method at its chosen k against the random control and against every feature."""
    references = {'random control, chosen k': chosen[chosen.selector == 'RandomSelector']}
    if len(everything):
        references[f'all {n_features} features'] = everything
    n_test, n_train = chosen.n_test.iloc[0], chosen.n_train.iloc[0]
    rows = []
    for against, reference in references.items():
        ref = reference.set_index(['task', 'split'])['roc_auc']
        for (task, name), part in chosen.groupby(['task', 'selector']):
            if name == 'RandomSelector':
                continue
            diff = (part.set_index(['task', 'split'])['roc_auc'] - ref).dropna()
            mean, low, high, p = corrected_test(diff, n_test, n_train)
            rows.append({'task': task, 'selector': name, 'against': against,
                         'mean_diff': mean, 'ci_low': low, 'ci_high': high, 'p_value': p,
                         'wins': int((diff > 0).sum()), 'splits': len(diff)})
    return pd.DataFrame(rows).sort_values(['against', 'task', 'mean_diff'],
                                          ascending=[True, True, False])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='+')
    parser.add_argument('--n-features', type=int, required=True,
                        help='Features in the data; random selection at this k is every feature')
    parser.add_argument('--out', required=True, help='Prefix of the three CSVs to write')
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    sequential = df.selector.str.startswith('Sequential')
    chosen = chosen_k(df[~sequential])
    everything = df[(df.selector == 'RandomSelector') & (df.k == args.n_features)]

    tables = {
        'chosen': chosen_table(chosen, args.n_features),
        'tests': tests_table(chosen, everything, args.n_features),
        'curve': df.groupby(['task', 'selector', 'k'])['roc_auc'].mean().unstack('k').reset_index(),
    }
    for name, table in tables.items():
        table.to_csv(f'{args.out}_{name}.csv', index=False)
        print(f'\n=== {name}\n' + table.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
