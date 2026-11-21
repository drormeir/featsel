"""
Test AUC against the number of features k: one method per family against
random features, one panel per task.

Reads the <out>_curve.csv tables that summarize_frontier.py writes (mean test
ROC AUC per task, selector and k) and runs nothing. Each panel shows random
selection (the bar), Mann-Whitney (a filter), tree importance (embedded) and
RFE (a wrapper), where the run has them, and a dotted line at the AUC of all
features. The gap between a method and random at a given k is what selection
adds there. With --mean, a dataset of several tasks gets one panel, the mean
over its tasks.

Usage:
    python scripts/plot_curves.py --mean --width 7.2 --out final_report/figures/curves.png \
        "SCAN-B=detailed_report/figures/summary_scanb70_curve.csv" \
        "BreastMNIST=detailed_report/figures/summary_breastmnist70_nocopies_curve.csv" \
        "FGVC-Aircraft=detailed_report/figures/summary_aircraft70_curve.csv"
    python scripts/plot_curves.py --out detailed_report/figures/curves_scanb70.png \
        "SCAN-B=detailed_report/figures/summary_scanb70_curve.csv"
    python scripts/plot_curves.py --out detailed_report/figures/curves_images70.png \
        "BreastMNIST=detailed_report/figures/summary_breastmnist70_nocopies_curve.csv" \
        "FGVC-Aircraft=detailed_report/figures/summary_aircraft70_curve.csv"
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
# Font sizes in points and line sizes: on screen, and at the printed width.
SCREEN = {'tick': 8, 'label': 9, 'title': 10, 'legend': 9, 'line': 2, 'marker': 6,
          'tick_gap': 0}
PRINT = {'tick': 6.5, 'label': 7, 'title': 8, 'legend': 7, 'line': 1.4, 'marker': 4,
         'tick_gap': 0.45}  # decades between labelled ticks, so labels do not touch


def panels(specs, mean=False):
    """
    (title, curve rows of one task) for every task of every 'label=csv' spec,
    or with mean, one panel per spec: the mean over its tasks.
    """
    for spec in specs:
        name, _, path = spec.partition('=')
        table = pd.read_csv(path)
        tasks = table.task.unique()
        if mean and len(tasks) > 1:
            rows = table.drop(columns='task').groupby('selector').mean()
            yield f'{name}: mean of {len(tasks)} tasks', rows.assign(task='mean')
            continue
        for task in tasks:
            title = name if len(tasks) == 1 else f'{name}: {task.removeprefix("ovr_")}'
            yield title, table[table.task == task].set_index('selector')


def draw(ax, title, rows, style):
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
        ax.plot(curve.index, curve.values, color=color, linewidth=style['line'],
                linestyle='--' if selector == RANDOM[0] else '-',
                marker=marker, markersize=style['marker'], markeredgecolor='white',
                markeredgewidth=1, label=name, zorder=3)
    # Named in the legend: a label on the line collides with the curves.
    ax.axhline(everything, color=MUTED, linewidth=1.2, linestyle=':', zorder=2,
               label='All features')
    ax.set_xscale('log')
    # 1,500 sits too close to 1,000 and 2,000 on a log axis to label.
    ticks = []
    for k in sorted(k for k in plotted if k != 1500):
        if not ticks or math.log10(k / ticks[-1]) >= style['tick_gap']:
            ticks.append(k)
    ax.set_xticks(ticks, [f'{int(k):,}' for k in ticks], fontsize=style['tick'])
    ax.minorticks_off()
    ax.tick_params(axis='y', labelsize=style['tick'], colors=MUTED)
    ax.tick_params(axis='x', colors=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.set_title(title, fontsize=style['title'], color=INK, loc='left')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('curves', nargs='+', help="'label=path' of a summary _curve.csv")
    parser.add_argument('--out', required=True, help='PNG to write')
    parser.add_argument('--mean', action='store_true',
                        help='one panel per dataset: the mean over its tasks')
    parser.add_argument('--width', type=float,
                        help='the printed width in inches; fonts keep their print size')
    args = parser.parse_args()
    style = PRINT if args.width else SCREEN

    found = list(panels(args.curves, args.mean))
    columns = min(3, len(found))
    rows = math.ceil(len(found) / columns)
    size = ((args.width, args.width * 0.42 * rows) if args.width
            else (4 * columns, 3.2 * rows + 0.6))
    fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    for ax, (title, table) in zip(axes.flat, found):
        draw(ax, title, table, style)
    for ax in axes.flat[len(found):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Test AUC, mean over splits', fontsize=style['label'], color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel('Features kept, k (log scale)', fontsize=style['label'], color=MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
               fontsize=style['legend'], bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9 if args.width else 0.94))
    fig.savefig(args.out, dpi=200 if args.width else 150, facecolor='#fcfcfb')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
