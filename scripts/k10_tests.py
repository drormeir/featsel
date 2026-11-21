"""
At a fixed k, one method against every other method, split by split, per task.

The main use is SCAN-B at k = 10: Mann-Whitney 1000 then RFE against the
other methods, Sequential selection included. Every runner CSV comes from a
config with the same seed and splits, so the methods pair by split. The test
is the corrected resampled t-test of Nadeau and Bengio (2003), as in
scripts/summarize_frontier.py.

Usage:
    python scripts/k10_tests.py results/local/frontier_scanb70_{gpu,cpu,chains,seq}.csv \
        --out detailed_report/figures/scanb70_k10_tests.csv
"""

import argparse

import pandas as pd
from summarize_frontier import corrected_test


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='+', help='runner CSVs on the same splits')
    parser.add_argument('--method', default='MW1000+RFE_newton')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True)
    df = df[df['k'] == args.k]
    n_test = df['n_test'].iloc[0]
    n_train = df['n_train'].iloc[0]

    rows = []
    for task, group in df.groupby('task'):
        per_split = group.pivot_table(index='split', columns='selector', values='roc_auc')
        for other in per_split.columns.drop(args.method):
            diff = (per_split[args.method] - per_split[other]).dropna()
            mean, low, high, p = corrected_test(diff, n_test, n_train)
            rows.append({'task': task, 'method': args.method, 'other': other,
                         'method_auc': per_split[args.method].mean(),
                         'other_auc': per_split[other].mean(),
                         'diff': mean, 'ci_low': low, 'ci_high': high, 'p_value': p,
                         'wins': int((diff > 0).sum()), 'splits': len(diff)})
    tests = pd.DataFrame(rows).sort_values(['task', 'diff'])
    tests.to_csv(args.out, index=False)
    print(tests.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
