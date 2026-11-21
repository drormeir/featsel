"""
HC2008 against every other method on the simulated rare signal, split by split.

Each method is taken at the k it chose on validation sets
(featsel.analysis.chosen_k). The test is the corrected resampled t-test of
Nadeau and Bengio (2003), as in scripts/summarize_frontier.py. Each row also
gives each method's mean recall and precision of the informative features.

Usage:
    python scripts/rare_signal_tests.py results/rare_signal70.csv \
        --out detailed_report/figures/paired_tests_rare_signal70.csv
"""

import argparse

import pandas as pd
from summarize_frontier import corrected_test

from featsel.analysis import chosen_k

REFERENCE = 'HC2008'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Random selection enters only at every feature, as the all-features row.
    random = df['selector'] == 'RandomSelector'
    df = df[~random | (df['k'] == df['n_selected'].max())]
    df.loc[df['selector'] == 'RandomSelector', 'selector'] = 'all features'
    chosen = chosen_k(df)
    per_split = chosen.groupby(['selector', 'split'])['roc_auc'].mean().unstack('selector')
    n_test = chosen['n_test'].iloc[0]
    n_train = chosen['n_train'].iloc[0]

    rows = []
    for other in per_split.columns.drop(REFERENCE):
        diff = (per_split[REFERENCE] - per_split[other]).dropna()
        mean, low, high, p = corrected_test(diff, n_test, n_train)
        rows.append({'method': REFERENCE, 'against': other, 'mean_diff': mean,
                     'ci_low': low, 'ci_high': high, 'p_value': p,
                     'wins': int((diff > 0).sum()), 'splits': len(diff)})
    tests = pd.DataFrame(rows)
    recovery = chosen.groupby('selector')[['roc_auc', 'n_selected', 'truth_recall',
                                           'truth_precision']].mean()
    tests = tests.join(recovery.add_prefix('against_'), on='against')
    tests.to_csv(args.out, index=False)
    print(tests.round(4).to_string(index=False))
    print('\n' + recovery.round(3).to_string())


if __name__ == '__main__':
    main()
