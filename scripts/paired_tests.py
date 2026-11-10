"""
Paired tests between selection methods, split by split.

Every method in a runner CSV ran on the same splits, so two methods can be
compared on each split. For each k this script takes each method's score on
each split (the mean over tasks within the split), and tests the paired
differences.

The main test is the corrected resampled t-test of Nadeau and Bengio (2003).
Random splits share most of their training samples, so the per-split scores
are not independent. The plain paired t-test, and the Wilcoxon signed-rank
test, then give p-values that are too small. The correction inflates the
variance of the mean difference from var / J to (1 / J + n_test / n_train)
* var, for J splits. The t statistic has J - 1 degrees of freedom.

For every method against the random control, and for every pair of methods,
the script reports the mean difference with its corrected 95% interval, how
many splits the first method won, the corrected p-value, and, for
comparison, the uncorrected Wilcoxon p-value.

Usage:
    python scripts/paired_tests.py results/comparison_scanb.csv --metric macro_f1
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

CONTROL = 'RandomSelector'


def per_split(results, metric):
    """One score per (selector, k, split): the mean over tasks and models."""
    return (results.groupby(['selector', 'k', 'split'], dropna=False)[metric].mean()
            .unstack('split'))


def corrected_t(diff, test_over_train):
    """Nadeau and Bengio's corrected resampled t-test: mean, 95% interval, p."""
    j = len(diff)
    mean = diff.mean()
    se = np.sqrt((1 / j + test_over_train) * diff.var(ddof=1))
    if se == 0:
        return mean, mean, mean, 1.0 if mean == 0 else 0.0
    half = student_t.ppf(0.975, j - 1) * se
    p = 2 * student_t.sf(abs(mean) / se, j - 1)
    return mean, mean - half, mean + half, p


def compare(scores, first, second, k, test_over_train):
    """Paired differences first - second at one k, over their common splits."""
    a, b = scores.loc[(first, k)], scores.loc[(second, k)]
    diff = (a - b).dropna()
    mean, low, high, p = corrected_t(diff, test_over_train)
    return {'k': k, 'first': first, 'second': second, 'splits': len(diff),
            'mean_diff': mean, 'ci_low': low, 'ci_high': high,
            'first_wins': int((diff > 0).sum()), 'p_value': p,
            'p_wilcoxon': wilcoxon(diff).pvalue if (diff != 0).any() else 1.0}


def tests(results, metric):
    scores = per_split(results, metric)
    test_over_train = (results['n_test'] / results['n_train']).mean()
    rows = []
    for k in sorted(results['k'].dropna().unique()):
        methods = [s for s in scores.index.get_level_values(0).unique()
                   if (s, k) in scores.index]
        if CONTROL in methods:
            rows += [compare(scores, m, CONTROL, k, test_over_train)
                     for m in methods if m != CONTROL]
        others = [m for m in methods if m != CONTROL]
        rows += [compare(scores, a, b, k, test_over_train)
                 for a, b in itertools.combinations(others, 2)]
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path, nargs='+')
    parser.add_argument('--metric', default='macro_f1')
    parser.add_argument('--out', type=Path, help='CSV to write the table to')
    args = parser.parse_args(argv)
    results = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True)
    table = tests(results, args.metric)
    if args.out:
        table.to_csv(args.out, index=False)
    with pd.option_context('display.width', 200):
        print(table.round({'mean_diff': 4, 'ci_low': 4, 'ci_high': 4,
                           'p_value': 6, 'p_wilcoxon': 6}).to_string(index=False))


if __name__ == '__main__':
    main()
