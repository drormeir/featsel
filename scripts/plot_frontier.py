"""
The cost-quality frontier: selection time against ROC AUC, one point per
selector and k, from one or more runner CSVs of the same splits.

Each point is the mean over every task and split of one (selector label, k)
cell: its median selection time on the x axis, log scale, and its mean
test ROC AUC on the y axis, with a bar of one standard deviation over the
splits. A point is on the frontier when no other point is both faster and
better. Frontier points are drawn in color and labelled; the rest are gray.

The machine and the code version come from the JSON the runner writes
beside each CSV, and go into the figure's caption: a time means something
only on the machine it was taken on.

One plot covers every task; each task then gets its own plot, named
<out>_<task>.png, with means over the splits only. Each plot has a CSV of
its table beside it.

Several CSVs are read as one table, such as a run of the GPU selectors and
a run of the CPU selectors; --out is then required.

Usage:
    python scripts/plot_frontier.py results/local/frontier_scanb70_gpu.csv
    python scripts/plot_frontier.py results/local/frontier_scanb70_gpu.csv \
        results/local/frontier_scanb70_cpu.csv --out results/local/frontier_scanb70.png
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


def place_labels(ax, labels):
    """
    Write each (x, y, text, size, color) label right of its point, top first,
    moving it down past any label already placed that it would overlap. A
    moved label gets a thin line back to its point.
    """
    points_per_pixel = 72 / ax.figure.dpi
    placed = []   # boxes in display pixels: (left, bottom, right, top)
    for x, y, text, size, color in sorted(labels, key=lambda item: -item[1]):
        px, py = ax.transData.transform((x, y))
        width = len(text) * size * 0.6 / points_per_pixel
        height = size * 1.3 / points_per_pixel
        # Frontier labels (INK) sit above their point: the frontier line runs
        # right from it.
        left = px + 6 / points_per_pixel
        bottom = py + (height * 0.3 if color == INK else -height / 2)
        while any(left < r and l < left + width and bottom < t and b < bottom + height
                  for l, b, r, t in placed):
            bottom -= height / 2
        placed.append((left, bottom, left + width, bottom + height))
        moved = abs(py - (bottom + height / 2)) > height
        ax.annotate(text, (x, y), xytext=((left - px) * points_per_pixel,
                                          (bottom + height / 2 - py) * points_per_pixel),
                    textcoords='offset points', fontsize=size, color=color, va='center',
                    arrowprops={'arrowstyle': '-', 'color': OTHER, 'linewidth': 0.6}
                    if moved else None)


def plot(table, on, metric, caption, out, title, over):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Each selector's k values joined in gray and named once, at its best
    # point, so a point off the frontier still says what it is.
    labels = []
    for selector, part in table.groupby('selector'):
        part = part.sort_values('k')
        ax.plot(part['selection_time_s'], part[metric], color=OTHER, linewidth=1,
                alpha=0.6, zorder=1)
        best = part.loc[part[metric].idxmax()]
        if not on[best.name]:
            labels.append((best['selection_time_s'], best[metric], short(selector), 7, MUTED))
    for mask, color, size in [(~on, OTHER, 22), (on, FRONTIER, 36)]:
        part = table[mask]
        ax.errorbar(part['selection_time_s'], part[metric], yerr=part[f'{metric}_std'],
                    fmt='none', ecolor=color, elinewidth=1, alpha=0.6)
        ax.scatter(part['selection_time_s'], part[metric], s=size, color=color,
                   edgecolor='white', linewidth=1.5, zorder=3)
    line = table[on].sort_values('selection_time_s')
    ax.step(line['selection_time_s'], line[metric], where='post', color=FRONTIER,
            linewidth=2, zorder=2)
    labels += [(row['selection_time_s'], row[metric], label(row), 8, INK)
               for _, row in line.iterrows()]

    ax.set_xscale('log')
    times = table['selection_time_s']
    ax.set_xlim(times.min() / 1.5, times.max() * 4)   # room for the rightmost label
    ax.set_xlabel('Selection time per fit, seconds (median, log scale)', color=INK)
    ax.set_ylabel(f'Test {metric.replace("_", " ").upper()} (mean over {over})', color=INK)
    ax.grid(True, which='major', color='#e3e5e8', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.set_title(title, loc='left', color=INK)
    fig.text(0.01, 0.01, caption, fontsize=7, color=MUTED, ha='left', va='bottom')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    place_labels(ax, labels)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path, nargs='+')
    parser.add_argument('--metric', default='roc_auc')
    parser.add_argument('--out', type=Path, help='PNG to write; default beside the CSV')
    args = parser.parse_args(argv)
    if len(args.csv) > 1 and args.out is None:
        parser.error('--out is required with more than one CSV')

    results = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True)

    runs = [json.loads(path.with_suffix('.json').read_text()) for path in args.csv]
    env = runs[0]['environment']
    commits = ', '.join(dict.fromkeys(
        f"{str(run['environment']['git_commit'])[:7]}"
        f"{' (modified)' if run['environment']['git_dirty'] else ''}" for run in runs))
    machine = f"{env['cpu']}, {env.get('gpu', 'GPU not recorded')}. Commit {commits}."

    out = args.out or args.csv[0].with_name(f"{args.csv[0].stem}_{args.metric}.png")
    # All tasks together, then each task on its own.
    parts = [(out, results, 'Feature selection: cost against quality', 'tasks and splits')]
    parts += [(out.with_name(f'{out.stem}_{task}.png'), part,
               f'Feature selection: cost against quality, {task}', 'splits')
              for task, part in results.groupby('task')]
    for path, part, title, over in parts:
        table = cells(part, args.metric)
        on = frontier(table, args.metric)
        caption = (f"{runs[0].get('dataset')}, {part['task'].nunique()} tasks x "
                   f"{part['split'].nunique()} splits, {part['model'].iloc[0]}. {machine}")
        plot(table, on, args.metric, caption, path, title, over)
        table.assign(on_frontier=on).to_csv(path.with_suffix('.csv'), index=False)
        print(f"Wrote {path} and its table {path.with_suffix('.csv')}")
        print(table.assign(on_frontier=on).sort_values('selection_time_s').to_string(index=False))


if __name__ == '__main__':
    main()
