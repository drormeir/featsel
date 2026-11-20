"""
How many differences the corrected resampled t-test establishes, against a
plain paired t-test, from the saved summary tables.

summarize_frontier.py writes each comparison's corrected p-value: the
variance of the mean difference is (1/J + n_test/n_train) times the sample
variance, for J splits. A plain paired t-test uses 1/J alone. So the plain
t statistic is the corrected one times sqrt((1/J + n_test/n_train) / (1/J)),
with the same J - 1 degrees of freedom. This script counts the comparisons
with p < 0.05 under each test, in the bar table (each method against random
at the same k) and the tests table (each method at its chosen k against all
features). It reads the CSVs and runs nothing.

Usage:
    python scripts/correction_effect.py --n-test 893 --n-train 2083 \
        detailed_report/figures/summary_scanb70
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def counts(table, n_test, n_train, alpha=0.05):
    """Comparisons, and those established by the corrected and the plain test."""
    j = table.splits
    t_corrected = stats.t.isf(table.p_value / 2, j - 1)
    t_plain = t_corrected * np.sqrt((1 / j + n_test / n_train) * j)
    p_plain = 2 * stats.t.sf(t_plain, j - 1)
    return len(table), int((table.p_value < alpha).sum()), int((p_plain < alpha).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('prefix', help='Prefix of the summary CSVs (<prefix>_bar.csv, _tests.csv)')
    parser.add_argument('--n-test', type=int, required=True)
    parser.add_argument('--n-train', type=int, required=True)
    args = parser.parse_args()

    rows = {name: counts(pd.read_csv(f'{args.prefix}_{name}.csv'), args.n_test, args.n_train)
            for name in ('bar', 'tests')}
    table = pd.DataFrame(rows, index=['comparisons', 'corrected', 'plain']).T
    table.loc['total'] = table.sum()
    print(table.to_string())


if __name__ == '__main__':
    main()
