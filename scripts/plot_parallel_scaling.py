"""
Plot the runner's speed-up by number of workers, from scripts/parallel_scaling.py.

One line: the median speed-up over one worker, with the range of the
repeats. A dashed line marks the ideal, speed-up equal to the workers. Runs
that lost workers are left out, as in parallel_scaling.py; the caption says
which worker counts they were.

Usage:
    python scripts/plot_parallel_scaling.py results/local/parallel_scaling.csv \
        --out final_report/figures/parallel_scaling.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

MEASURED = '#2a78d6'  # the reference palette's first categorical hue, as in plot_frontier.py
IDEAL = '#9aa0a6'
INK = '#1f2328'
MUTED = '#5f6368'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.csv)
    valid = table[table.workers == table.ended_with_workers]
    lost = sorted(set(table.workers) - set(valid.workers))
    reference = valid.loc[valid.workers == 1, 'seconds'].median()
    speed = (reference / valid.seconds).groupby(valid.workers).agg(['median', 'min', 'max'])

    fig, ax = plt.subplots(figsize=(6, 3.8))
    top = table.workers.max()
    ax.plot([1, top], [1, top], linestyle='--', color=IDEAL, linewidth=1.5)
    ax.text(top - 0.1, top - 0.35, 'ideal', color=MUTED, fontsize=9, ha='right')
    ax.vlines(speed.index, speed['min'], speed['max'], color=MEASURED, linewidth=2)
    ax.plot(speed.index, speed['median'], color=MEASURED, linewidth=2, marker='o',
            markersize=8, markeredgecolor='white', markeredgewidth=2)
    for workers, value in speed['median'].items():
        ax.annotate(f'{value:.2f}', (workers, value), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=8, color=INK)

    ax.set_xlim(0.6, top + 0.4)
    ax.set_ylim(0.5, top + 0.5)
    ax.set_xticks(range(1, top + 1))
    ax.set_xlabel('Workers', color=INK)
    ax.set_ylabel('Speed-up over one worker', color=INK)
    ax.grid(True, color='#e3e5e8', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.set_title('The runner on full SCAN-B, by number of workers', loc='left', color=INK)
    caption = (f'Lasso and RFE at k = 10, 50, 250; 5 splits x 5 tasks = 50 jobs. '
               f'One worker: {reference:.0f} s. Median of {valid.groupby("workers").size().min()} runs; '
               f'bars: range.')
    if lost:
        caption += (f'\nNot shown: {", ".join(map(str, lost))} workers lost workers '
                    'to the memory cap in every run.')
    fig.text(0.01, 0.01, caption, fontsize=7, color=MUTED, ha='left', va='bottom')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
