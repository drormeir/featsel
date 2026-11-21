"""
A chain's first-step size: Mann-Whitney keeping 250 or 1,000 features
against the 500 of the main runs, split by split.

Each chain is taken at the k it chose on validation sets
(featsel.analysis.chosen_k). Its time is the mean over splits of the
median selection time per fit at that k. The test is the corrected resampled t-test of
Nadeau and Bengio (2003), as in scripts/summarize_frontier.py. Both CSVs
come from configs with the same seed and splits.

Usage:
    python scripts/chain_size_tests.py results/local/frontier_breastmnist70_chains.csv \
        results/local/chain_size_breastmnist70.csv \
        --out detailed_report/figures/chain_size_breastmnist70.csv
"""

import argparse

import pandas as pd
from summarize_frontier import corrected_test

from featsel.analysis import chosen_k

REFERENCE = 500


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('main_csv', help='the run with the 500-feature chains')
    parser.add_argument('sweep_csv', help='the run with the other first-step sizes')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(args.main_csv), pd.read_csv(args.sweep_csv)])
    df = df[df['selector'].str.match(r'MW\d+\+')]
    parts = df['selector'].str.extract(r'MW(\d+)\+(.+)')
    df = df.assign(first=parts[0].astype(int), second=parts[1])
    chosen = chosen_k(df)
    n_test = chosen['n_test'].iloc[0]
    n_train = chosen['n_train'].iloc[0]

    rows = []
    for second, group in chosen.groupby('second'):
        per_split = group.pivot_table(index='split', columns='first', values='roc_auc')
        for first in per_split.columns:
            rows_first = group[group['first'] == first]
            row = {'second_step': second, 'first_step': first,
                   'mean_auc': per_split[first].mean(),
                   'median_k': rows_first['n_selected'].median(),
                   'mean_time_s': rows_first['selection_time_s'].mean()}
            if first != REFERENCE:
                diff = (per_split[first] - per_split[REFERENCE]).dropna()
                mean, low, high, p = corrected_test(diff, n_test, n_train)
                row.update({'diff_vs_500': mean, 'ci_low': low, 'ci_high': high,
                            'p_value': p, 'wins': int((diff > 0).sum()),
                            'splits': len(diff)})
            rows.append(row)
    tests = pd.DataFrame(rows)
    tests.to_csv(args.out, index=False)
    print(tests.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
