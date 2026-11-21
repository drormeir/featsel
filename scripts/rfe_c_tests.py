"""
RFE's SVM penalty: C = 0.001, 0.1 and 1 against the 0.01 of the main runs,
split by split.

Each setting is taken at the k it chose on validation sets
(featsel.analysis.chosen_k). With --every-k, the sweep ran without a k
search, and each C is compared with 0.01 at every k of the list instead:
the k is then fixed, so the test set chooses nothing. The test is the corrected resampled t-test of
Nadeau and Bengio (2003), as in scripts/summarize_frontier.py. Both CSVs
come from configs with the same seed and splits; on SCAN-B each of the
five tasks is tested on its own.

Usage:
    python scripts/rfe_c_tests.py results/local/frontier_breastmnist70_cpu.csv \
        results/local/rfe_c_breastmnist70.csv \
        --out detailed_report/figures/rfe_c_breastmnist70.csv
    python scripts/rfe_c_tests.py results/local/frontier_scanb70_cpu.csv \
        results/local/rfe_c_scanb70.csv --every-k \
        --out detailed_report/figures/rfe_c_scanb70.csv
"""

import argparse

import pandas as pd
from summarize_frontier import corrected_test

from featsel.analysis import chosen_k

REFERENCE = 0.01


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('main_csv', help='the run with RFE at C = 0.01 (label RFE_newton)')
    parser.add_argument('sweep_csv', help='the run with the other values of C')
    parser.add_argument('--every-k', action='store_true',
                        help='compare at every k, not at the chosen k')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    main_rows = pd.read_csv(args.main_csv)
    main_rows = main_rows[main_rows['selector'] == 'RFE_newton'].assign(C=REFERENCE)
    sweep = pd.read_csv(args.sweep_csv)
    sweep = sweep.assign(C=sweep['selector'].str.extract(r'RFE_C(.+)')[0].astype(float))
    both = pd.concat([main_rows, sweep])
    chosen = both if args.every_k else chosen_k(both)
    n_test = chosen['n_test'].iloc[0]
    n_train = chosen['n_train'].iloc[0]
    by = ['task', 'k'] if args.every_k else ['task']

    rows = []
    for key, group in chosen.groupby(by):
        per_split = group.pivot_table(index='split', columns='C', values='roc_auc')
        for c in per_split.columns:
            at_c = group[group['C'] == c]
            row = {**dict(zip(by, key)), 'C': c, 'mean_auc': per_split[c].mean(),
                   'median_k': at_c['n_selected'].median()}
            if c != REFERENCE:
                diff = (per_split[c] - per_split[REFERENCE]).dropna()
                mean, low, high, p = corrected_test(diff, n_test, n_train)
                row.update({'diff_vs_0.01': mean, 'ci_low': low, 'ci_high': high,
                            'p_value': p, 'wins': int((diff > 0).sum()),
                            'splits': len(diff)})
            rows.append(row)
    tests = pd.DataFrame(rows)
    tests.to_csv(args.out, index=False)
    print(tests.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
