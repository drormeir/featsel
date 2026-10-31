"""
Summarize a selector-comparison CSV from the runner as Markdown tables.

Each table cell is the median over splits, with the range over splits in
brackets: 5 splits are too few for quartiles (docs/experiment_design.md 2a).
Scores over tasks are the mean of the tasks within each split.

Usage:
    python scripts/summarize_comparison.py results/comparison_scanb.csv \
        --n-features 30865 --out docs/comparison_scanb.md
"""

import argparse
import json

import pandas as pd

from featsel.analysis import kuncheva_index


def _cell(values):
    """Median with the range over splits, as text."""
    return f'{values.median():.3f} [{values.min():.3f}, {values.max():.3f}]'


def _k_label(k):
    return 'own' if pd.isna(k) else str(int(k))


def _markdown(table):
    """A DataFrame as a Markdown table, its index as the first column."""
    header = [table.index.name or '', *map(str, table.columns)]
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for label, row in table.iterrows():
        lines.append('| ' + ' | '.join([str(label), *map(str, row)]) + ' |')
    return '\n'.join(lines)


def _table(frame, index, columns, values):
    """A Markdown table of median [min, max] text cells."""
    table = frame.groupby([index, columns], observed=True)[values].apply(_cell)
    return _markdown(table.unstack(columns).fillna(''))


def summarize(df, n_features):
    stability = kuncheva_index(df, n_features)
    selectors = list(dict.fromkeys(df.selector))
    labels = [*sorted({_k_label(k) for k in df.k} - {'own'}, key=int), 'own']
    df = df.assign(k=pd.Categorical(df.k.map(_k_label), labels, ordered=True),
                   selector=pd.Categorical(df.selector, selectors, ordered=True))
    sections = []

    over_tasks = df.groupby(['selector', 'k', 'split'], observed=True).macro_f1.mean()
    over_tasks = over_tasks.reset_index()
    sections.append(f'## Macro-F1, mean over the {df.task.nunique()} tasks\n\n'
                    + _table(over_tasks, 'selector', 'k', 'macro_f1'))

    random = over_tasks[over_tasks.selector == 'RandomSelector'].set_index(['k', 'split'])
    gain = over_tasks[over_tasks.k != 'own'].join(random.macro_f1.rename('random'),
                                                  on=['k', 'split'])
    gain['gain'] = gain.macro_f1 - gain.random
    sections.append('## Gain in macro-F1 over RandomSelector, same k and split\n\n'
                    + _table(gain[gain.selector != 'RandomSelector'],
                             'selector', 'k', 'gain'))

    for k in labels[:-1]:
        sections.append(f'## Macro-F1 per task, k = {k}\n\n'
                        + _table(df[df.k == k], 'selector', 'task', 'macro_f1'))
    sections.append('## Macro-F1 per task, own count (Higher Criticism)\n\n'
                    + _table(df[df.k == 'own'], 'selector', 'task', 'macro_f1'))
    sections.append('## Genes chosen by Higher Criticism, per task\n\n'
                    + _table(df[df.k == 'own'], 'selector', 'task', 'n_selected'))

    selection = df.drop_duplicates(['task', 'selector', 'k', 'split'])
    sections.append('## Selection time in seconds, per fit\n\n'
                    + _table(selection, 'selector', 'k', 'selection_time_s'))

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
