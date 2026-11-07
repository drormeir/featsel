"""
The cost-quality frontier: selection time against ROC AUC, one point per
selector and k, from a runner CSV.

Each point is the mean over every task and split of one (selector label, k)
cell: its median selection time on the x axis, log scale, and its mean
test ROC AUC on the y axis, with a bar of one standard deviation over the
splits. A point is on the frontier when no other point is both faster and
better. Frontier points are drawn in color and labelled; the rest are gray.

The machine and the code version come from the JSON the runner writes
beside its CSV, and go into the figure's caption: a time means something
only on the machine it was taken on.

Usage:
    python scripts/plot_frontier.py results/local/frontier_scanb.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

FRONTIER = '#2a78d6'     # the reference palette's first categorical hue
OTHER = '#9aa0a6'
INK = '#1f2328'
MUTED = '#5f6368'


def cells(results, metric):
    """One row per (selector, k): median time, mean and spread of the metric."""
    # Time is a property of one fit: the same for every classifier of a cell.
    fits = results.drop_duplicates(['task', 'split', 'selector', 'k'])
    time = fits.groupby(['selector', 'k'], dropna=False)['selection_time_s'].median()
    per_split = results.groupby(['selector', 'k', 'split'], dropna=False)[metric].mean()
    quality = per_split.groupby(['selector', 'k'], dropna=False).agg(['mean', 'std'])
    table = quality.join(time).reset_index()
    return table.rename(columns={'mean': metric, 'std': f'{metric}_std'})


def frontier(table, metric):
    """Mask of the points no other point beats in both time and the metric."""
    order = table.sort_values(['selection_time_s', metric], ascending=[True, False])
    best = -float('inf')
    on = pd.Series(False, index=table.index)
    for index, row in order.iterrows():
        if row[metric] > best:
            on[index] = True
            best = row[metric]
    return on


def label(row):
    """'Selector k=50', or the selector alone when it picks its own count."""
    name = short(row['selector'])
    return name if pd.isna(row['k']) else f"{name} k={int(row['k'])}"


def short(selector):
    """The selector's name without the 'Selector' every class name ends in."""
    return selector.removesuffix('Selector')


def plot(table, on, metric, caption, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Each selector's k values joined in gray and named once, at its best
    # point, so a point off the frontier still says what it is.
    for selector, part in table.groupby('selector'):
        part = part.sort_values('k')
        ax.plot(part['selection_time_s'], part[metric], color=OTHER, linewidth=1,
                alpha=0.6, zorder=1)
        best = part.loc[part[metric].idxmax()]
        if not on[best.name]:
            ax.annotate(short(selector), (best['selection_time_s'], best[metric]),
                        xytext=(6, 4), textcoords='offset points', fontsize=7, color=MUTED)
    for mask, color, size in [(~on, OTHER, 22), (on, FRONTIER, 36)]:
        part = table[mask]
        ax.errorbar(part['selection_time_s'], part[metric], yerr=part[f'{metric}_std'],
                    fmt='none', ecolor=color, elinewidth=1, alpha=0.6)
        ax.scatter(part['selection_time_s'], part[metric], s=size, color=color,
                   edgecolor='white', linewidth=1.5, zorder=3)
    line = table[on].sort_values('selection_time_s')
    ax.step(line['selection_time_s'], line[metric], where='post', color=FRONTIER,
            linewidth=2, zorder=2)
    for _, row in line.iterrows():
        ax.annotate(label(row), (row['selection_time_s'], row[metric]),
                    xytext=(6, -12), textcoords='offset points', fontsize=8, color=INK)

    ax.set_xscale('log')
    ax.set_xlabel('Selection time per fit, seconds (median, log scale)', color=INK)
    ax.set_ylabel(f'Test {metric.replace("_", " ").upper()} (mean over tasks and splits)',
                  color=INK)
    ax.grid(True, which='major', color='#e3e5e8', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.set_title('Feature selection: cost against quality', loc='left', color=INK)
    fig.text(0.01, 0.01, caption, fontsize=7, color=MUTED, ha='left', va='bottom')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--metric', default='roc_auc')
    parser.add_argument('--out', type=Path, help='PNG to write; default beside the CSV')
    args = parser.parse_args(argv)

    results = pd.read_csv(args.csv)
    table = cells(results, args.metric)
    on = frontier(table, args.metric)

    details = json.loads(args.csv.with_suffix('.json').read_text())
    env = details['environment']
    caption = (f"{details.get('dataset')}, {results['task'].nunique()} tasks x "
               f"{results['split'].nunique()} splits, {results['model'].iloc[0]}. "
               f"{env['cpu']}, {env.get('gpu', 'no GPU')}. "
               f"Commit {str(env['git_commit'])[:7]}{' (modified)' if env['git_dirty'] else ''}.")

    out = args.out or args.csv.with_name(f"{args.csv.stem}_{args.metric}.png")
    plot(table, on, args.metric, caption, out)
    table.assign(on_frontier=on).to_csv(out.with_suffix('.csv'), index=False)
    print(f"Wrote {out} and its table {out.with_suffix('.csv')}")
    print(table.assign(on_frontier=on).sort_values('selection_time_s').to_string(index=False))


if __name__ == '__main__':
    main()
