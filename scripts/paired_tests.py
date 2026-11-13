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

Methods that choose their own count, such as Higher Criticism, have no k.
With --own-count they are tested against each other, and against every
method at the largest k.

Usage:
    python scripts/paired_tests.py results/comparison_scanb.csv --metric macro_f1
    python scripts/paired_tests.py results/comparison_breastmnist.csv \
        results/comparison_breastmnist_all_features.csv --baseline AllFeatures
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


def compare(a, b, k, first, second, test_over_train):
    """Paired differences a - b, per-split scores of first and second, over common splits."""
    diff = (a - b).dropna()
    mean, low, high, p = corrected_t(diff, test_over_train)
    return {'k': k, 'first': first, 'second': second, 'splits': len(diff),
            'mean_diff': mean, 'ci_low': low, 'ci_high': high,
            'first_wins': int((diff > 0).sum()), 'p_value': p,
            'p_wilcoxon': wilcoxon(diff).pvalue if (diff != 0).any() else 1.0}


def tests(results, metric, baseline=None, own_count=False):
    """
    Every method against the random control and every pair of methods, at
    each k. With a baseline, a selector label such as all features, every
    method at every k is tested against that baseline instead. With
    own_count, the methods without a k are tested against each other and
    against every method at the largest k.
    """
    scores = per_split(results, metric)
    test_over_train = (results['n_test'] / results['n_train']).mean()
    if baseline is not None:
        # A baseline with one setting, such as all features, meets every
        # method at every k. A baseline with a k list, such as random, meets
        # each method at the same k only.
        reference = scores.loc[baseline]
        single = len(reference) == 1
        return pd.DataFrame([
            compare(row, reference.iloc[0] if single else reference.loc[k], k, method, baseline,
                    test_over_train)
            for (method, k), row in scores.drop(baseline, level=0).iterrows()
            if single or k in reference.index])
    rows = []
    for k in sorted(results['k'].dropna().unique()):
        methods = [s for s in scores.index.get_level_values(0).unique()
                   if (s, k) in scores.index]
        pairs = [(m, CONTROL) for m in methods if m != CONTROL] if CONTROL in methods else []
        pairs += itertools.combinations([m for m in methods if m != CONTROL], 2)
        rows += [compare(scores.loc[(a, k)], scores.loc[(b, k)], k, a, b, test_over_train)
                 for a, b in pairs]
    if own_count:
        largest = results['k'].max()
        own = [m for m, k in scores.index if pd.isna(k)]
        others = [m for m, k in scores.index if k == largest]
        rows += [compare(scores.loc[a].iloc[0], scores.loc[b].iloc[0], float('nan'), a, b,
                         test_over_train) for a, b in itertools.combinations(own, 2)]
        rows += [compare(scores.loc[a].iloc[0], scores.loc[(b, largest)], largest, a, b,
                         test_over_train) for a in own for b in others]
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path, nargs='+')
    parser.add_argument('--metric', default='macro_f1')
    parser.add_argument('--out', type=Path, help='CSV to write the table to')
    parser.add_argument('--baseline', help='Selector label to test every method against, '
                                           'such as all features')
    parser.add_argument('--own-count', action='store_true',
                        help='Also test the methods that choose their own count')
    args = parser.parse_args(argv)
    results = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True)
    table = tests(results, args.metric, args.baseline, args.own_count)
    if args.out:
        table.to_csv(args.out, index=False)
    with pd.option_context('display.width', 200):
        print(table.round({'mean_diff': 4, 'ci_low': 4, 'ci_high': 4,
                           'p_value': 6, 'p_wilcoxon': 6}).to_string(index=False))


if __name__ == '__main__':
    main()
