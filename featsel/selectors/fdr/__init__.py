"""
False discovery rate: keep features whose F-test p-values pass Benjamini-Hochberg.
"""

import numpy as np

from ..base import FeatureSelector
from ..f_test import f_test


class FDRSelector(FeatureSelector):
    """
    Select features by F-test p-value, controlling the false discovery rate.

    Computes one p-value per feature with the ANOVA F-test, then applies the
    Benjamini-Hochberg procedure: on average at most a fraction alpha of the
    kept features are false discoveries. Picks its own feature count.

    Applies scikit-learn's SelectFdr rule to the F-test of f_test, which runs
    on the CPU or a GPU.

    Best for: a principled count when many features carry signal
    Pros: fast, a clear statistical guarantee, adapts the count to the data
    Cons: same blind spots as the F-test; can keep thousands of features
    on a strong target, or none on a weak one

    Parameters
    ----------
    alpha : float, default=0.05
        Target false discovery rate.
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
    >>> from featsel.selectors import FDRSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_features=50, n_informative=5,
    ...                            n_redundant=0, shuffle=False, random_state=0)
    >>> # Features 0-4 are informative.
    >>> FDRSelector(alpha=0.01).fit(X, y).get_support(indices=True).tolist()
    [0, 1, 2, 4]
    """

    def __init__(self, alpha=0.05, task='classification', device='auto',
                 check_input=True):
        self.alpha = alpha
        self.task = task
        self.device = device
        self.check_input = check_input

    def _fit(self, X, y):
        """
        Keep every feature the Benjamini-Hochberg procedure accepts at alpha.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : FDRSelector
            Fitted selector.
        """
        X, y = self._validate(X, y, keep_gpu_tensor=True, check_input=self.check_input)

        self.scores_, self.pvalues_, self.device_ = f_test(X, y, self.task, self.device)

        # Benjamini-Hochberg, as SelectFdr: the largest p-value under its
        # rank's share of alpha, and every p-value up to it.
        n_features = len(self.pvalues_)
        ordered = np.sort(self.pvalues_)
        passing = ordered[ordered <= self.alpha / n_features * np.arange(1, n_features + 1)]
        self.support_ = (self.pvalues_ <= passing.max() if passing.size
                         else np.zeros(n_features, dtype=bool))

        if not np.any(self.support_):
            raise ValueError(f"No feature passes the false discovery rate at alpha={self.alpha}. "
                             "Increase alpha.")
        return self
