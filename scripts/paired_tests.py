"""
Paired tests between selection methods, split by split.

Every method in a runner CSV ran on the same splits, so two methods can be
compared on each split. For each k this script takes each method's score on
each split (the mean over tasks within the split), and tests the paired
differences with the Wilcoxon signed-rank test (two-sided). It reports the
median difference, how many splits the first method won, and the p-value,
for every method against the random control and for every pair of methods.

With n splits the smallest possible two-sided p-value is 2 / 2**n: 3e-5 for
16 splits, 0.0625 for 5. So 5 splits cannot show a difference at 5%.

Usage:
    python scripts/paired_tests.py results/comparison_scanb.csv --metric macro_f1
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

CONTROL = 'RandomSelector'


def per_split(results, metric):
    """One score per (selector, k, split): the mean over tasks and models."""
    return (results.groupby(['selector', 'k', 'split'], dropna=False)[metric].mean()
            .unstack('split'))


def compare(scores, first, second, k):
    """Paired differences first - second at one k, over their common splits."""
    a, b = scores.loc[(first, k)], scores.loc[(second, k)]
    diff = (a - b).dropna()
    p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
    return {'k': k, 'first': first, 'second': second, 'splits': len(diff),
            'median_diff': diff.median(), 'first_wins': int((diff > 0).sum()),
            'p_value': p}


def tests(results, metric):
    scores = per_split(results, metric)
    rows = []
    for k in sorted(results['k'].dropna().unique()):
        methods = [s for s in scores.index.get_level_values(0).unique()
                   if (s, k) in scores.index]
        if CONTROL in methods:
            rows += [compare(scores, m, CONTROL, k) for m in methods if m != CONTROL]
        others = [m for m in methods if m != CONTROL]
        rows += [compare(scores, a, b, k) for a, b in itertools.combinations(others, 2)]
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
        print(table.round({'median_diff': 4, 'p_value': 6}).to_string(index=False))


if __name__ == '__main__':
    main()
