"""
False discovery rate: keep features whose F-test p-values pass Benjamini-Hochberg.
"""

import numpy as np
from sklearn.feature_selection import SelectFdr, f_classif, f_regression

from ..base import FeatureSelector


class FDRSelector(FeatureSelector):
    """
    Select features by F-test p-value, controlling the false discovery rate.

    Computes one p-value per feature with the ANOVA F-test, then applies the
    Benjamini-Hochberg procedure: on average at most a fraction alpha of the
    kept features are false discoveries. Picks its own feature count.

    Wraps scikit-learn's SelectFdr.

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
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    scores_ : np.ndarray
        F statistic for each feature.
    pvalues_ : np.ndarray
        p-value for each feature.
    feature_importances_ : np.ndarray
        Alias for scores_.

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

    def __init__(self, alpha=0.05, task='classification', check_input=True):
        self.alpha = alpha
        self.task = task
        self.check_input = check_input

    def fit(self, X, y):
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
        X, y = self._validate(X, y)

        if self.task == 'classification':
            score_func = f_classif
        elif self.task == 'regression':
            score_func = f_regression
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        fdr = SelectFdr(score_func, alpha=self.alpha).fit(X, y)
        self.scores_ = fdr.scores_
        self.pvalues_ = fdr.pvalues_
        self.feature_importances_ = self.scores_
        self.support_ = fdr.get_support()

        if not np.any(self.support_):
            raise ValueError(f"No feature passes the false discovery rate at alpha={self.alpha}. "
                             "Increase alpha.")
        return self
