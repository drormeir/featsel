"""
Cost against quality, one panel per dataset: the final report's Figure 1.

Reads the frontier tables that plot_frontier.py writes (one row per method
and k: mean test ROC AUC and median selection time per fit) and runs
nothing. Seven methods have their own color, in the reference palette's
fixed order; random selection is gray diamonds; every other method is a
small gray dot. The dark step line is the frontier: no point is both faster
and better than a point on it.

Usage:
    python scripts/plot_final_frontier.py --out final_report/figures/frontier.png \
        "SCAN-B, mean of 5 tasks=detailed_report/figures/frontier_scanb70_roc_auc.csv" \
        "BreastMNIST=detailed_report/figures/frontier_breastmnist70_roc_auc.csv" \
        "FGVC-Aircraft=detailed_report/figures/frontier_aircraft70_roc_auc.csv"
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# The reference palette's first seven categorical hues, in fixed order. The
# first three match plot_curves.py, so a method keeps its color across figures.
SERIES = [
    ('MannWhitneySelector', 'Mann-Whitney', '#2a78d6', 'o'),
    ('RFE_newton', 'RFE', '#eb6834', '^'),
    ('TreeImportanceSelector', 'Tree importance', '#1baf7a', 's'),
    ('MutualInfoSelector', 'Mutual information', '#eda100', 'v'),
    ('Lasso_ppdna', 'Lasso', '#e87ba4', 'P'),
    ('HC2008', 'HC2008', '#008300', 'h'),
    ('BerkJones', 'Berk-Jones', '#6250d6', 'd'),
]
RANDOM = ('RandomSelector', 'Random', '#898781', 'D')
INK = '#0b0b0b'
MUTED = '#52514e'
OTHER = '#c3c2b7'
GRID = '#e1e0d9'


def draw(ax, title, table):
    """One dataset: every point, the highlighted methods, and the frontier."""
    named = {RANDOM[0], *(s[0] for s in SERIES)}
    rest = table[~table.selector.isin(named)]
    ax.scatter(rest.selection_time_s, rest.roc_auc, s=8, color=OTHER, zorder=2,
               label='Other methods')
    # Markers only: random's points form most of the frontier's cheap end, so
    # a line of its own would lie on the frontier.
    random = table[table.selector == RANDOM[0]]
    ax.scatter(random.selection_time_s, random.roc_auc, s=18, color=RANDOM[2],
               marker=RANDOM[3], edgecolors='white', linewidths=0.8, label=RANDOM[1],
               zorder=3)
    for selector, name, color, marker in SERIES:
        rows = table[table.selector == selector]
        if rows.empty:
            continue
        # A thin ring: a thick one hides the thin crossed markers (Lasso).
        ax.scatter(rows.selection_time_s, rows.roc_auc, s=30, color=color, marker=marker,
                   edgecolors='white', linewidths=0.4, label=name, zorder=4)
    front = table[table.on_frontier].sort_values('selection_time_s')
    ax.step(front.selection_time_s, front.roc_auc, where='post', color=INK,
            linewidth=1.2, label='Frontier', zorder=5)
    ax.set_xscale('log')
    ax.tick_params(labelsize=6.5, colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(OTHER)
    ax.set_title(title, fontsize=8, color=INK, loc='left')
    ax.set_xlabel('Selection time per fit, s (log)', fontsize=7, color=MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('frontiers', nargs='+', help="'label=path' of a frontier CSV")
    parser.add_argument('--out', required=True, help='PNG to write')
    args = parser.parse_args()

    # Drawn at the printed width of an A4 page, so the text prints at its size.
    fig, axes = plt.subplots(1, len(args.frontiers), figsize=(7.2, 3.0), squeeze=False)
    for ax, spec in zip(axes[0], args.frontiers):
        title, _, path = spec.partition('=')
        draw(ax, title, pd.read_csv(path))
    axes[0, 0].set_ylabel('Test AUC, mean over splits', fontsize=7, color=MUTED)
    # The first panel has every series; aircraft lacks tree importance.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=0.6)
    fig.savefig(args.out, dpi=200, facecolor='#fcfcfb')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
