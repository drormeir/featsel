"""
liblinear against the exact solvers, on SCAN-B, before liblinear was dropped.

The source is the first 70% frontier run of 2026-11-09 to 11, archived in
results/local/archive_2026-11-13/ when liblinear was dropped (section 6.2).
Its CPU run holds RFE with liblinear and with Newton, and Lasso with
liblinear, on 5 splits; its GPU run holds Lasso with PPDNA, on 16. The
runs cannot be made again: the package no longer has liblinear.

For each pair (RFE: liblinear, Newton; Lasso: liblinear, PPDNA), on the
splits both hold, the table gives each solver's mean test AUC over tasks
and splits at each k, the median selection time per fit, and on how many
splits the second solver scored higher. No test is run: 5 splits are too
few.

Usage:
    python scripts/liblinear_vs_exact.py results/local/archive_2026-11-13 \
        --out detailed_report/figures/liblinear_vs_exact.csv
"""

import argparse
from pathlib import Path

import pandas as pd

PAIRS = [('RFE', 'RFE_liblinear', 'RFE_newton'), ('Lasso', 'Lasso_liblinear', 'Lasso_ppdna')]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    runs = pd.concat([pd.read_csv(args.archive / f'frontier_scanb70_{name}.csv')
                      for name in ('cpu', 'gpu')], ignore_index=True)
    rows = []
    for method, old, exact in PAIRS:
        both = runs[runs.selector.isin([old, exact])]
        splits = set(both[both.selector == old].split) & set(both[both.selector == exact].split)
        both = both[both.split.isin(splits)]
        per_split = both.groupby(['k', 'selector', 'split']).roc_auc.mean().unstack('selector')
        for k, part in per_split.groupby(level='k'):
            rows.append({'method': method, 'k': k, 'splits': len(part),
                         'liblinear_auc': part[old].mean(), 'exact_auc': part[exact].mean(),
                         'exact_wins': int((part[exact] > part[old]).sum()),
                         'liblinear_s_per_fit': both[both.selector == old].selection_time_s.median(),
                         'exact_s_per_fit': both[both.selector == exact].selection_time_s.median()})
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
