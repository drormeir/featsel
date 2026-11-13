"""
Aggregation of the runner's results.

Every function takes the long-format DataFrame read from a runner CSV: one
row per (task, split, train size, preprocessing, selector, k, model) execution.
"""

import numpy as np
import pandas as pd

from .metrics import KEY_COLUMNS, METRICS, SCORE_METRICS

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
    columns = [c for c in [*METRICS, *SCORE_METRICS, *COST_COLUMNS] if c in results.columns]
    grouped = results.groupby(CELL_COLUMNS, dropna=False)[columns]

    summary = pd.concat({
        'median': grouped.median(),
        'q1': grouped.quantile(0.25),
        'q3': grouped.quantile(0.75),
    }, axis=1)
    summary.columns = [f'{metric}_{stat}' for stat, metric in summary.columns]
    return summary.reset_index()


def chosen_k(results):
    """
    One row per (task, split, train size, preprocessing, selector, model):
    the k with the best validation ROC AUC, chosen without the test set.

    Needs the 'validation_roc_auc' column, which the runner writes when the
    config sets k_validation. A tie goes to the smaller k. A selector that
    chooses its own count has one row per split and no validation score, so
    its row passes through. The row keeps its test-set metrics, so a mean
    over splits of its roc_auc estimates the AUC of the method with k chosen
    on validation sets.
    """
    keys = [c for c in KEY_COLUMNS if c != 'k']
    ordered = results.assign(_k=results['k'].fillna(-1)).sort_values(
        [*keys, 'validation_roc_auc', '_k'], ascending=[True] * len(keys) + [False, True],
        na_position='last')
    return ordered.drop_duplicates(keys).drop(columns='_k').reset_index(drop=True)


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


def nogueira_stability(results, n_total_features):
    """
    Selection stability across splits, for subsets of any size.

    The estimator of Nogueira, Sechidis and Brown (2018):

        phi = 1 - mean_f(s_f^2) / ((k / n) * (1 - k / n))

    where s_f^2 = M / (M - 1) * p_f * (1 - p_f) is the unbiased variance of
    feature f's selection over the M splits, p_f the share of splits that
    selected it, k the mean subset size and n the total number of features.
    It is 1 when every split selects the same features, and 0 for selection
    at random. Unlike Kuncheva's index it does not assume equal subset sizes;
    when they are equal, it equals the mean pairwise Kuncheva index.

    Parameters
    ----------
    results : pd.DataFrame
        Runner output, with a space-separated 'selected_indices' column.
    n_total_features : int
        Total number of features available to the selector.

    Returns
    -------
    stability : pd.DataFrame
        One row per selection cell with the stability over its splits.
    """
    selections = results.drop_duplicates([*SELECTION_COLUMNS, 'split'])
    rows = []

    for key, group in selections.groupby(SELECTION_COLUMNS, dropna=False):
        subsets = [set(s.split()) for s in group['selected_indices'].fillna('').astype(str)]
        m = len(subsets)
        share = np.mean([len(s) for s in subsets]) / n_total_features
        if m < 2 or not 0 < share < 1:
            continue

        counts = pd.Series([f for s in subsets for f in s]).value_counts()
        p = counts.to_numpy() / m
        mean_variance = m / (m - 1) * (p * (1 - p)).sum() / n_total_features
        rows.append({
            **dict(zip(SELECTION_COLUMNS, key)),
            'stability': float(1 - mean_variance / (share * (1 - share))),
            'n_splits': m,
            'variable_size': len({len(s) for s in subsets}) > 1,
        })

    return pd.DataFrame(rows)
