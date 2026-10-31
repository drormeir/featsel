"""
Summarize a selector-comparison CSV from the runner as Markdown tables.

Each table cell is the median over splits, with the first and third
quartiles in brackets (docs/experiment_design.md 2a). Scores over tasks are
the mean of the tasks within each split.

Usage:
    python scripts/summarize_comparison.py results/comparison_scanb.csv \
        --n-features 30865 --out docs/comparison_scanb.md
"""

import argparse
import json

import numpy as np
import pandas as pd

from featsel.analysis import kuncheva_index


def _cell(values, digits=3):
    """Median with the first and third quartiles over splits, as text."""
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return f'{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]'


def _k_label(k):
    return 'own' if pd.isna(k) else str(int(k))


def _markdown(table):
    """A DataFrame as a Markdown table, its index as the first column."""
    header = [table.index.name or '', *map(str, table.columns)]
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for label, row in table.iterrows():
        lines.append('| ' + ' | '.join([str(label), *map(str, row)]) + ' |')
    return '\n'.join(lines)


def _table(frame, index, columns, values, digits=3):
    """A Markdown table of median [min, max] text cells."""
    table = frame.groupby([index, columns], observed=True)[values].apply(_cell, digits=digits)
    return _markdown(table.unstack(columns).fillna(''))


def _own_count_gains(df):
    """
    Each own-count method against fixed-k methods at its own size.

    For every task and split, a method without k (Higher Criticism) is
    compared with each fixed-k method run at the k nearest its feature
    count, if that k is within 1% of it. Returns rows of (selector,
    against, split, gain), the gain averaged over tasks.
    """
    fixed = df[df.k.notna()]
    ks = np.sort(fixed.k.unique())
    rows = []
    for _, own in df[df.k.isna()].iterrows():
        k = ks[np.argmin(np.abs(ks - own.n_selected))]
        if abs(k - own.n_selected) > 0.01 * own.n_selected:
            continue
        same = fixed[(fixed.task == own.task) & (fixed.split == own.split) & (fixed.k == k)]
        rows += [{'selector': own.selector, 'against': f'{other.selector}, k = {int(k)}',
                  'task': own.task, 'split': own.split,
                  'gain': own.macro_f1 - other.macro_f1} for _, other in same.iterrows()]
    gains = pd.DataFrame(rows, columns=['selector', 'against', 'task', 'split', 'gain'])
    return gains.groupby(['selector', 'against', 'split'], as_index=False).gain.mean()


def _ordered(df):
    """k as text labels ('own' last) and selectors in config order, both ordered."""
    selectors = list(dict.fromkeys(df.selector))
    labels = [*sorted({_k_label(k) for k in df.k} - {'own'}, key=int), 'own']
    return df.assign(k=pd.Categorical(df.k.map(_k_label), labels, ordered=True),
                     selector=pd.Categorical(df.selector, selectors, ordered=True)), labels


def _over_tasks(df):
    """Macro-F1 per (selector, k, split), the mean over tasks."""
    over = _ordered(df)[0].groupby(['selector', 'k', 'split'], observed=True).macro_f1.mean()
    return over.reset_index()


def macro_f1_table(df):
    """Macro-F1 by selector and k, mean over tasks, median [range] over splits."""
    return _table(_over_tasks(df), 'selector', 'k', 'macro_f1')


def gain_table(df):
    """Macro-F1 above RandomSelector at the same k and split."""
    over = _over_tasks(df)
    random = over[over.selector == 'RandomSelector'].set_index(['k', 'split'])
    gain = over[over.k != 'own'].join(random.macro_f1.rename('random'), on=['k', 'split'])
    gain['gain'] = gain.macro_f1 - gain.random
    return _table(gain[gain.selector != 'RandomSelector'], 'selector', 'k', 'gain')


def own_count_table(df):
    """Own-count methods against fixed-k methods at their size; '' if none match."""
    gains = _own_count_gains(df)
    return _table(gains, 'selector', 'against', 'gain') if len(gains) else ''


def summarize(df, n_features):
    stability = kuncheva_index(df, n_features)
    own_count = own_count_table(df)
    sections = [f'## Macro-F1, mean over the {df.task.nunique()} tasks\n\n' + macro_f1_table(df),
                '## Gain in macro-F1 over RandomSelector, same k and split\n\n' + gain_table(df)]
    if own_count:
        sections.append('## Own-count methods against fixed-k methods at the same size\n\n'
                        'Macro-F1 of the own-count method minus that of the fixed-k method '
                        'at the k nearest its feature count, same task and split.\n\n'
                        + own_count)
    df, labels = _ordered(df)
    selectors = list(df.selector.cat.categories)

    for k in labels[:-1]:
        sections.append(f'## Macro-F1 per task, k = {k}\n\n'
                        + _table(df[df.k == k], 'selector', 'task', 'macro_f1'))
    sections.append('## Macro-F1 per task, own count (Higher Criticism)\n\n'
                    + _table(df[df.k == 'own'], 'selector', 'task', 'macro_f1'))
    selection = df.drop_duplicates(['task', 'selector', 'k', 'split'])
    sections.append('## Features selected, over tasks and splits\n\n'
                    'Below `k` when a method keeps fewer: Lasso keeps only non-zero '
                    'weights.\n\n'
                    + _table(selection, 'selector', 'k', 'n_selected', digits=0))
    sections.append('## Features chosen by Higher Criticism, per task\n\n'
                    + _table(df[df.k == 'own'], 'selector', 'task', 'n_selected', digits=0))
    sections.append('## Selection time in seconds, per fit\n\n'
                    'Rank-once methods (Mann-Whitney, Lasso, tree importance, RFE) fit '
                    'once per task and split; that time is shown under every `k`.\n\n'
                    + _table(selection, 'selector', 'k', 'selection_time_s', digits=1))

    stability = stability.assign(k=stability.k.map(_k_label))
    stability = stability.groupby(['selector', 'k']).consistency_index.mean().unstack('k')
    stability = stability.reindex(index=selectors, columns=labels).round(3).fillna('')
    sections.append("## Kuncheva's stability index, mean over tasks\n\n"
                    + _markdown(stability))

    chosen = df.tuned_params.map(lambda p: json.loads(p).get('C')).rename('C')
    counts = chosen.groupby(df.k, observed=True).value_counts().unstack(fill_value=0)
    sections.append('## Chosen C, count of cells\n\n' + _markdown(counts))

    return '\n\n'.join(sections) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='Runner output CSV')
    parser.add_argument('--n-features', type=int, required=True,
                        help='Features available to the selectors, for the stability index')
    parser.add_argument('--out', required=True, help='Markdown file to write')
    args = parser.parse_args()

    body = summarize(pd.read_csv(args.csv), args.n_features)
    with open(args.out, 'w') as f:
        f.write(body)


if __name__ == '__main__':
    main()
