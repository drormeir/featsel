"""
Family-wise error rate: keep features whose F-test p-values pass Bonferroni.
"""

import numpy as np

from ..base import FeatureSelector
from ..f_test import f_test


class FWESelector(FeatureSelector):
    """
    Select features by F-test p-value, controlling the family-wise error rate.

    Computes one p-value per feature with the ANOVA F-test and keeps those
    below alpha divided by the number of features (Bonferroni). The chance of
    keeping even one uninformative feature is then at most alpha. Picks its
    own feature count.

    Applies scikit-learn's SelectFwe rule to the F-test of f_test, which runs
    on the CPU or a GPU.

    Best for: a short list where every feature should be real
    Pros: fast, the strictest guarantee of the p-value selectors
    Cons: same blind spots as the F-test; with thousands of features the
    cutoff is so strict it can keep very few, or none

    Parameters
    ----------
    alpha : float, default=0.05
        Family-wise error rate.
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
    >>> from featsel.selectors import FWESelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_features=50, n_informative=5,
    ...                            n_redundant=0, shuffle=False, random_state=0)
    >>> # Features 0-4 are informative. Bonferroni is stricter than FDR.
    >>> FWESelector(alpha=0.01).fit(X, y).get_support(indices=True).tolist()
    [1, 2]
    """

    def __init__(self, alpha=0.05, task='classification', device='auto',
                 check_input=True):
        self.alpha = alpha
        self.task = task
        self.device = device
        self.check_input = check_input

    def _fit(self, X, y):
        """
        Keep every feature with p-value below alpha / n_features.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : FWESelector
            Fitted selector.
        """
        X, y = self._validate(X, y, keep_gpu_tensor=True)

        self.scores_, self.pvalues_, self.device_ = f_test(X, y, self.task, self.device)
        self.support_ = self.pvalues_ < self.alpha / len(self.pvalues_)   # Bonferroni, as SelectFwe

        if not np.any(self.support_):
            raise ValueError(f"No feature passes the family-wise error rate at "
                             f"alpha={self.alpha}. Increase alpha.")
        return self
