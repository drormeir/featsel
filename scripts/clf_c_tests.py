"""
The classifier's L2 penalty: logistic regression at other values of C
against the C = 1 of the main runs, split by split.

One run scores every C on the same selections and splits; the classifiers
are labelled lr_C<value>. Each (method, C) is taken at the k it chose on
validation sets (featsel.analysis.chosen_k). The test is the corrected
resampled t-test of Nadeau and Bengio (2003), as in
scripts/summarize_frontier.py; each task is tested on its own.

Usage:
    python scripts/clf_c_tests.py results/local/clf_c_breastmnist70.csv \
        --out detailed_report/figures/clf_c_breastmnist70.csv
"""

import argparse

import pandas as pd
from summarize_frontier import corrected_test

from featsel.analysis import chosen_k

REFERENCE = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', help='the run with every value of C')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.assign(C=df['model'].str.extract(r'lr_C(.+)')[0].astype(float))
    chosen = chosen_k(df)
    n_test = chosen['n_test'].iloc[0]
    n_train = chosen['n_train'].iloc[0]

    rows = []
    for (task, method), group in chosen.groupby(['task', 'selector']):
        per_split = group.pivot_table(index='split', columns='C', values='roc_auc')
        for c in per_split.columns:
            row = {'task': task, 'method': method, 'C': c,
                   'mean_auc': per_split[c].mean(),
                   'median_k': group[group['C'] == c]['n_selected'].median()}
            if c != REFERENCE:
                diff = (per_split[c] - per_split[REFERENCE]).dropna()
                mean, low, high, p = corrected_test(diff, n_test, n_train)
                row.update({'diff_vs_1': mean, 'ci_low': low, 'ci_high': high,
                            'p_value': p, 'wins': int((diff > 0).sum()),
                            'splits': len(diff)})
            rows.append(row)
    tests = pd.DataFrame(rows)
    tests.to_csv(args.out, index=False)
    print(tests.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
