"""
Aggregation of the runner's results.

Both functions take the long-format DataFrame read from a runner CSV: one row
per (task, split, train size, preprocessing, selector, k, model) execution.
"""

import numpy as np
import pandas as pd

from .metrics import KEY_COLUMNS, METRICS

# One experiment cell: every key except the split, which varies inside it.
CELL_COLUMNS = [c for c in KEY_COLUMNS if c != 'split']

# One selection: the model does not affect which features were chosen.
SELECTION_COLUMNS = [c for c in CELL_COLUMNS if c != 'model']

COST_COLUMNS = ['n_selected', 'selection_time_s', 'selection_peak_mb', 'fit_time_s']


def summarize(results):
    """
    Median, first and third quartile over splits, one row per cell.

    The median and interquartile range describe split-to-split variation
    without assuming it is symmetric.
    """
    columns = [c for c in [*METRICS, *COST_COLUMNS] if c in results.columns]
    grouped = results.groupby(CELL_COLUMNS, dropna=False)[columns]

    summary = pd.concat({
        'median': grouped.median(),
        'q1': grouped.quantile(0.25),
        'q3': grouped.quantile(0.75),
    }, axis=1)
    summary.columns = [f'{metric}_{stat}' for stat, metric in summary.columns]
    return summary.reset_index()


def kuncheva_index(results, n_total_features):
    """
    Selection stability across splits, corrected for chance agreement.

    Kuncheva's consistency index for a pair of equally sized subsets:

        I = (r - k^2 / n) / (k - k^2 / n)

    where r is the size of the intersection, k the subset size and n the total
    number of features. It is 1 for identical subsets and 0 for the overlap
    expected by chance, which is why a random selector scores near 0 here even
    though its raw Jaccard overlap is positive. Reported as the mean over all
    split pairs.

    Parameters
    ----------
    results : pd.DataFrame
        Runner output, with a space-separated 'selected_indices' column.
    n_total_features : int
        Total number of features available to the selector.

    Returns
    -------
    stability : pd.DataFrame
        One row per selection cell with the mean pairwise consistency index.
    """
    selections = results.drop_duplicates([*SELECTION_COLUMNS, 'split'])
    rows = []

    for key, group in selections.groupby(SELECTION_COLUMNS, dropna=False):
        subsets = [set(s.split()) for s in group['selected_indices'].fillna('').astype(str)]
        if len(subsets) < 2:
            continue

        pair_scores = []
        for i in range(len(subsets)):
            for j in range(i + 1, len(subsets)):
                # Use the actual subset sizes; they match unless a selector
                # picks its own feature count (e.g. Higher Criticism).
                k_eff = (len(subsets[i]) + len(subsets[j])) / 2
                expected = k_eff ** 2 / n_total_features
                denominator = k_eff - expected
                if denominator <= 0:
                    continue
                r = len(subsets[i] & subsets[j])
                pair_scores.append((r - expected) / denominator)

        if pair_scores:
            rows.append({
                **dict(zip(SELECTION_COLUMNS, key)),
                'consistency_index': float(np.mean(pair_scores)),
                'n_pairs': len(pair_scores),
                'variable_size': len({len(s) for s in subsets}) > 1,
            })

    return pd.DataFrame(rows)
