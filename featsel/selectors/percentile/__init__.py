"""
Percentile: keep the top percentage of features by F statistic.
"""

import numpy as np

from ..base import FeatureSelector
from ..f_test import f_test


class PercentileSelector(FeatureSelector):
    """
    Select the top percentage of features by ANOVA F statistic.

    The same ranking as ANOVAFSelector, but the count is a share of the input
    features instead of a fixed number. Useful when datasets of different
    widths should keep the same share.

    Applies scikit-learn's SelectPercentile rule to the F-test of f_test, which runs
    on the CPU or a GPU.

    Best for: comparing datasets with different feature counts
    Pros: fast, scale-free cutoff
    Cons: same blind spots as the F-test; the count grows with the input
    even when the signal does not

    Parameters
    ----------
    percentile : float, default=10
        Percentage of features to keep, from 0 to 100.
    task : str, default='classification'
        'classification' uses f_classif, 'regression' uses f_regression.
    device : str, default='auto'
        Where the F-test's sums run: 'auto' uses a CUDA GPU when one is free
        (the `gpu` extra), else NumPy; 'cpu' always scikit-learn's own
        function; 'cuda' always the GPU, waiting while another fit holds it.
        A tensor on a GPU runs there whatever this says. The GPU sums in
        float64 and rounds differently, so near-ties can order differently.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    device_ : str
        The device the F-test ran on: 'cpu' or 'cuda'.
    scores_ : np.ndarray
        F statistic for each feature.
    pvalues_ : np.ndarray
        p-value for each feature.

    Examples
    --------
    >>> from featsel.selectors import PercentileSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=40, n_informative=10)
    >>> PercentileSelector(percentile=25).fit_transform(X, y).shape
    (100, 10)
    """

    def __init__(self, percentile=10, task='classification', device='auto',
                 check_input=True):
        self.percentile = percentile
        self.task = task
        self.device = device
        self.check_input = check_input

    def _fit(self, X, y):
        """
        Keep the highest-scoring percentile of features.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : PercentileSelector
            Fitted selector.
        """
        X, y = self._validate(X, y, keep_gpu_tensor=True)

        self.scores_, self.pvalues_, self.device_ = f_test(X, y, self.task, self.device)
        self.support_ = self._percentile_mask()
        return self

    def _percentile_mask(self):
        """SelectPercentile's rule: scores above the cut, then ties at it up to the count."""
        if self.percentile == 100:
            return np.ones(len(self.scores_), dtype=bool)
        if self.percentile == 0:
            return np.zeros(len(self.scores_), dtype=bool)
        scores = np.nan_to_num(self.scores_, nan=np.finfo(self.scores_.dtype).min)
        threshold = np.percentile(scores, 100 - self.percentile)
        mask = scores > threshold
        ties = np.flatnonzero(scores == threshold)
        mask[ties[:int(len(scores) * self.percentile / 100) - mask.sum()]] = True
        return mask
