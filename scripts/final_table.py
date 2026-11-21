"""
Each method at the k it chose on validation sets, on all three datasets: the
final report's Table 5.

Reads the _chosen.csv tables that summarize_frontier.py writes and runs
nothing. SCAN-B's five tasks are averaged: test AUC, selection time per fit
(each task's median) and stability (Nogueira's measure). Writes the table as
CSV and as Markdown, ready to include. The CSV also holds each method's
interquartile range of test AUC over the splits (on SCAN-B, the mean over
its tasks), for the caption.

Usage:
    python scripts/final_table.py --out final_report/figures/table5 \
        --scanb detailed_report/figures/summary_scanb70_chosen.csv \
        --breastmnist detailed_report/figures/summary_breastmnist70_nocopies_chosen.csv \
        --aircraft detailed_report/figures/summary_aircraft70_chosen.csv
"""

import argparse
from pathlib import Path

import pandas as pd

# The run's labels, by prefix, and the names the report uses. A chain's
# first stage keeps 1,000 genes on SCAN-B and 500 features on BreastMNIST.
NAMES = {
    'MutualInfoSelector': 'Mutual information',
    'BerkJones': 'Berk-Jones',
    'HC2008': 'HC2008',
    'CorrelationSelector': 'Correlation (Spearman)',
    'MannWhitneySelector': 'Mann-Whitney',
    'TreeImportanceSelector': 'Tree importance',
    'MW+Lasso_ppdna': 'Mann-Whitney, then Lasso',
    'MW+TreeImportance': 'Mann-Whitney, then tree importance',
    'MW+RFE_newton': 'Mann-Whitney, then RFE',
    'Lasso_ppdna': 'Lasso',
    'all features': 'All features (no selection)',
    'RFECV_newton_auc': 'RFECV, scored by AUC',
    'RFE_newton': 'RFE',
    'HC2004': 'HC2004',
    'RFECV_newton': 'RFECV, scored by accuracy',
}


def name(selector):
    """The report's name for a run label such as 'MW1000+RFE_newton'."""
    if selector.startswith('MW') and '+' in selector:
        selector = 'MW+' + selector.split('+', 1)[1]
    if selector.startswith('all '):
        selector = 'all features'
    return NAMES[selector]


def per_method(path):
    """Mean over the tasks of one dataset, per method."""
    table = pd.read_csv(path)
    table['method'] = table.selector.map(name)
    table['roc_auc_iqr'] = table.roc_auc_q3 - table.roc_auc_q1
    return table.groupby('method')[
        ['roc_auc', 'roc_auc_iqr', 'selection_time_s', 'stability']].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scanb', required=True)
    parser.add_argument('--breastmnist', required=True,
                        help='BreastMNIST without the copied images')
    parser.add_argument('--aircraft', required=True)
    parser.add_argument('--out', required=True, type=Path,
                        help='output path without suffix; writes .csv and .md')
    args = parser.parse_args()

    scanb = per_method(args.scanb)
    breastmnist = per_method(args.breastmnist)
    aircraft = per_method(args.aircraft)
    table = pd.DataFrame({
        'SCAN-B AUC': scanb.roc_auc,
        'BreastMNIST AUC': breastmnist.roc_auc,
        'FGVC-Aircraft AUC': aircraft.roc_auc,
        'SCAN-B selection, s per fit': scanb.selection_time_s,
        'SCAN-B stability': scanb.stability,
        'SCAN-B AUC IQR': scanb.roc_auc_iqr,
        'BreastMNIST AUC IQR': breastmnist.roc_auc_iqr,
        'FGVC-Aircraft AUC IQR': aircraft.roc_auc_iqr,
    }).sort_values('SCAN-B AUC', ascending=False)
    # All features selects nothing: no selection time, no stability.
    table.loc[NAMES['all features'], 'SCAN-B selection, s per fit'] = None
    table.index.name = 'Method'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out.with_suffix('.csv'))

    shown = table.drop(columns=[c for c in table if c.endswith('IQR')])
    for column in shown:
        digits = 2 if 'stability' in column else 3
        if 'per fit' in column:
            shown[column] = shown[column].map(
                lambda t: '—' if pd.isna(t) else f'{t:.0f}' if t >= 10 else f'{t:#.2g}')
        else:
            shown[column] = shown[column].map(
                lambda v, d=digits: f'{v:.{d}f}' if pd.notna(v) else '—')
    # Short headers, broken into lines, leave the method column room; the
    # caption says the first three are test AUC.
    shown.columns = ['SCAN-B', 'BreastMNIST', 'FGVC-<br>Aircraft', 'SCAN-B,<br>s per fit',
                     'SCAN-B<br>stability']
    lines = ['| Method | ' + ' | '.join(shown.columns) + ' |',
             '|---' * (len(shown.columns) + 1) + '|']
    lines += [f'| {method} | ' + ' | '.join(row) + ' |' for method, row in shown.iterrows()]
    markdown = '\n'.join(lines) + '\n'
    args.out.with_suffix('.md').write_text(markdown)
    print(markdown)


if __name__ == '__main__':
    main()
