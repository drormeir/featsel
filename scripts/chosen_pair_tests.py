"""
Every pair of methods, each at the k it chose on validation sets, per task.

summarize_frontier.py tests each method at its chosen k only against all
features and against random features. This script tests the methods against
each other: on each split, each method's test AUC at the k it chose there
(featsel.analysis.chosen_k), and the paired differences over the splits. The
test is the corrected resampled t-test of Nadeau and Bengio (2003), from
paired_tests.py. Sequential selection and the random control are left out.

The table has one row per task and pair. The printout counts the pairs with
p < 0.05, without and with a Bonferroni correction over the pairs, for the
methods named with --methods (default: all).

Usage:
    python scripts/chosen_pair_tests.py results/local/frontier_scanb70_{gpu,cpu,chains,berkjones,rfecv_auc}.csv \
        --out detailed_report/figures/chosen_pairs_scanb70.csv
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
from paired_tests import corrected_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from featsel.analysis import chosen_k


def pair_tests(results):
    """One row per task and pair of methods, at each method's chosen k."""
    drop = results.selector.str.startswith('Sequential') | (results.selector == 'RandomSelector')
    chosen = chosen_k(results[~drop])
    test_over_train = (results['n_test'] / results['n_train']).mean()
    rows = []
    for task, part in chosen.groupby('task'):
        scores = part.groupby(['selector', 'split'])['roc_auc'].mean().unstack('split')
        for first, second in itertools.combinations(scores.index, 2):
            diff = (scores.loc[first] - scores.loc[second]).dropna()
            mean, low, high, p = corrected_t(diff, test_over_train)
            rows.append({'task': task, 'first': first, 'second': second,
                         'first_auc': scores.loc[first].mean(),
                         'second_auc': scores.loc[second].mean(),
                         'mean_diff': mean, 'ci_low': low, 'ci_high': high,
                         'first_wins': int((diff > 0).sum()), 'splits': len(diff),
                         'p_value': p})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='+', help='runner CSVs on the same splits')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--methods', nargs='+', help='run labels to count pairs among')
    parser.add_argument('--within', type=float,
                        help='count pairs among the methods whose mean AUC over tasks is '
                             'within this distance of the best')
    args = parser.parse_args()

    table = pair_tests(pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True))
    table.to_csv(args.out, index=False)
    methods = args.methods
    if args.within is not None:
        auc = pd.concat([table.set_index(['task', 'first']).first_auc,
                         table.set_index(['task', 'second']).second_auc])
        mean = auc[~auc.index.duplicated()].groupby(level=1).mean().sort_values(ascending=False)
        methods = list(mean.index[mean >= mean.iloc[0] - args.within])
        print(mean[methods].round(4).to_string())
    among = table
    if methods:
        among = table[table['first'].isin(methods) & table['second'].isin(methods)]
    print(f'pairs: {len(among)}, p < 0.05: {(among.p_value < 0.05).sum()}, '
          f'Bonferroni (p < 0.05 / {len(among)}): {(among.p_value < 0.05 / len(among)).sum()}')
    print(among[among.p_value < 0.05].groupby('task').size().to_string())


if __name__ == '__main__':
    main()
