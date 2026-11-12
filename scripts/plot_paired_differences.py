"""
Plot each method's paired difference from one reference method, at each k,
from a table written by scripts/paired_tests.py.

One panel per k. Each row is a method: a dot at its mean difference from
the reference over the splits, and a bar for its corrected interval. The
interval is Bonferroni-adjusted for every test in the table, as chapter 7
of the final report requires: confidence 1 - 0.05 / m for m tests. A bar
that does not cross zero is drawn in color: the difference passes
p < 0.05 / m. A bar that crosses zero is gray. The random control is left
out; its gap is far larger than the gaps between methods.

Usage:
    python scripts/plot_paired_differences.py \
        final_report/figures/paired_tests_frontier_scanb70_16_splits.csv \
        --reference MannWhitneySelector --out final_report/figures/frontier_scanb70_16_splits.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from paired_tests import CONTROL
from plot_frontier import short
from scipy.stats import t as student_t

ESTABLISHED = '#2a78d6'  # the reference palette's first categorical hue, as in plot_frontier.py
OPEN = '#9aa0a6'
INK = '#1f2328'
MUTED = '#5f6368'
NAMES = {'Lasso_ppdna': 'Lasso, PPDNA', 'MannWhitneySelector': 'Mann-Whitney',
         'TreeImportanceSelector': 'Tree importance', 'MutualInfoSelector': 'Mutual information',
         'CorrelationSelector': 'Spearman correlation', 'BerkJones': 'Berk-Jones'}


def name(method):
    """The method's name as the report writes it."""
    return NAMES.get(method, short(method))


def bonferroni(table):
    """The table with each 95% interval widened to 1 - 0.05 / m, m its number of tests."""
    dof = table.splits - 1
    se = (table.ci_high - table.mean_diff) / student_t.ppf(0.975, dof)
    half = student_t.ppf(1 - 0.025 / len(table), dof) * se
    return table.assign(ci_low=table.mean_diff - half, ci_high=table.mean_diff + half)


def against(table, reference):
    """Rows of every method against the reference, signed as method - reference."""
    first = table[table.second == reference].rename(columns={'first': 'method'})
    flipped = table[table['first'] == reference].rename(columns={'second': 'method'})
    flipped = flipped.assign(mean_diff=-flipped.mean_diff, ci_low=-flipped.ci_high,
                             ci_high=-flipped.ci_low)
    rows = pd.concat([first, flipped], ignore_index=True)
    return rows[rows.method != CONTROL][['k', 'method', 'mean_diff', 'ci_low', 'ci_high', 'splits']]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--metric', default='AUC', help='Name of the metric, for the axis')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.csv)
    rows = against(bonferroni(table), args.reference)
    methods = sorted(rows.method.unique(), key=name)
    ks = sorted(rows.k.unique())
    limit = max(rows.ci_high.abs().max(), rows.ci_low.abs().max()) * 1.1

    fig, axes = plt.subplots(1, len(ks), figsize=(2.4 * len(ks) + 1, 3), sharey=True)
    for ax, k in zip(axes, ks):
        panel = rows[rows.k == k].set_index('method').reindex(methods)
        for y, (method, row) in enumerate(panel.iterrows()):
            color = OPEN if row.ci_low <= 0 <= row.ci_high else ESTABLISHED
            ax.hlines(y, row.ci_low, row.ci_high, color=color, linewidth=2)
            ax.plot(row.mean_diff, y, 'o', color=color, markersize=8,
                    markeredgecolor='white', markeredgewidth=2)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_xlim(-limit, limit)
        ax.set_title(f'k = {int(k)}', color=INK, fontsize=10)
        ax.grid(True, axis='x', color='#e3e5e8', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8, left=False)
    axes[0].set_yticks(range(len(methods)), [name(m) for m in methods], color=INK)
    axes[0].invert_yaxis()
    fig.supxlabel(f'{args.metric} minus {name(args.reference)}, paired over splits',
                  color=INK, fontsize=9, y=0.13)
    fig.suptitle(f'Each method against {name(args.reference)} at the same k',
                 x=0.01, ha='left', color=INK)
    caption = (f'Dot: mean difference over {int(rows.splits.min())} splits. Bar: corrected '
               f'interval (Nadeau and Bengio), Bonferroni-adjusted for {len(table)} tests. '
               f'Blue: it excludes zero, p < 0.05 / {len(table)}. Gray: it does not.')
    own = sorted({name(m) for m in table.loc[table.k.isna(), ['first', 'second']].values.ravel()})
    if own:
        caption += (f'\n{" and ".join(own)} choose their own count; they are compared with '
                    f'{name(args.reference)} at k = {int(ks[-1])}.')
    fig.text(0.01, 0.01, caption, fontsize=7, color=MUTED, ha='left', va='bottom')
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
