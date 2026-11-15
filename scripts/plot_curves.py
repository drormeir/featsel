"""
Test AUC against the number of features k: one method per family against
random features, one panel per task.

Reads the <out>_curve.csv tables that summarize_frontier.py writes (mean test
ROC AUC per task, selector and k) and runs nothing. Each panel shows random
selection (the bar), Mann-Whitney (a filter), tree importance (embedded) and
RFE (a wrapper), where the run has them, and a dotted line at the AUC of all
features. The gap between a method and random at a given k is what selection
adds there.

Usage:
    python scripts/plot_curves.py --out final_report/figures/curves_scanb70.png \
        "SCAN-B=final_report/figures/summary_scanb70_curve.csv"
    python scripts/plot_curves.py --out final_report/figures/curves_images70.png \
        "BreastMNIST=final_report/figures/summary_breastmnist70_curve.csv" \
        "FGVC-Aircraft=final_report/figures/summary_aircraft70_curve.csv"
"""

import argparse
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# The reference palette's first three categorical hues, in fixed order.
# Random is the bar, so it is gray and dashed, not a categorical hue.
SERIES = [
    ('MannWhitneySelector', 'Mann-Whitney', '#2a78d6', 'o'),
    ('TreeImportanceSelector', 'Tree importance', '#1baf7a', 's'),
    ('RFE_newton', 'RFE', '#eb6834', '^'),
]
RANDOM = ('RandomSelector', 'Random', '#898781', 'D')
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e1e0d9'


def panels(specs):
    """(title, curve rows of one task) for every task of every 'label=csv' spec."""
    for spec in specs:
        name, _, path = spec.partition('=')
        table = pd.read_csv(path)
        tasks = table.task.unique()
        for task in tasks:
            title = name if len(tasks) == 1 else f'{name}: {task.removeprefix("ovr_")}'
            yield title, table[table.task == task].set_index('selector')


def draw(ax, title, rows):
    """One task: each series against k, random dashed, all features dotted."""
    ks = [float(c) for c in rows.columns if c != 'task']
    everything = rows.loc[RANDOM[0], str(max(ks))]
    plotted = set()
    for selector, name, color, marker in [RANDOM, *SERIES]:
        if selector not in rows.index:
            continue
        curve = rows.loc[selector].drop('task').astype(float)
        curve.index = curve.index.astype(float)
        curve = curve[(curve.index < max(ks)) & curve.notna()]
        plotted |= set(curve.index)
        ax.plot(curve.index, curve.values, color=color, linewidth=2,
                linestyle='--' if selector == RANDOM[0] else '-',
                marker=marker, markersize=6, markeredgecolor='white',
                markeredgewidth=1, label=name, zorder=3)
    # Named in the legend: a label on the line collides with the curves.
    ax.axhline(everything, color=MUTED, linewidth=1.2, linestyle=':', zorder=2,
               label='All features')
    ax.set_xscale('log')
    # 1,500 sits too close to 1,000 and 2,000 on a log axis to label.
    ticks = sorted(k for k in plotted if k != 1500)
    ax.set_xticks(ticks, [f'{int(k):,}' for k in ticks], fontsize=8)
    ax.minorticks_off()
    ax.tick_params(axis='y', labelsize=8, colors=MUTED)
    ax.tick_params(axis='x', colors=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.set_title(title, fontsize=10, color=INK, loc='left')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('curves', nargs='+', help="'label=path' of a summary _curve.csv")
    parser.add_argument('--out', required=True, help='PNG to write')
    args = parser.parse_args()

    found = list(panels(args.curves))
    columns = min(3, len(found))
    rows = math.ceil(len(found) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.2 * rows + 0.6),
                             squeeze=False)
    for ax, (title, table) in zip(axes.flat, found):
        draw(ax, title, table)
    for ax in axes.flat[len(found):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Test AUC, mean over splits', fontsize=9, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel('Features kept, k (log scale)', fontsize=9, color=MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150, facecolor='#fcfcfb')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
