"""
ANOVA F-test: rank features by their univariate F statistic.
"""

import numpy as np
from sklearn.feature_selection import f_classif, f_regression

from ..base import FeatureSelector


class ANOVAFSelector(FeatureSelector):
    """
    Select features based on ANOVA F-statistic (classification) or F-test (regression).

    Best for: Linear relationships, quick univariate screening
    Pros: Very fast (O(n*m)), interpretable (p-values), well-established statistical theory
    Cons: Only captures linear relationships, assumes feature independence

    Parameters
    ----------
    n_features : int, optional
        Number of top features to select. If None, uses alpha threshold.
    alpha : float, default=0.05
        Significance level for feature selection (used if n_features is None).
        Features with p-value < alpha are selected.
    task : str, default='classification'
        Type of task: 'classification' or 'regression'.
        Determines whether to use f_classif or f_regression.

    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    scores_ : np.ndarray
        F-statistic for each feature.
    pvalues_ : np.ndarray
        p-value for each feature.
    feature_importances_ : np.ndarray
        Alias for scores_.

    Examples
    --------
    >>> from featsel.selectors import ANOVAFSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> ANOVAFSelector(n_features=10).fit_transform(X, y).shape
    (100, 10)
    """

    def __init__(self, n_features=None, alpha=0.05, task='classification', check_input=True):
        self.n_features = n_features
        self.alpha = alpha
        self.task = task
        self.check_input = check_input

    def fit(self, X, y):
        """
        Fit the selector on training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ANOVAFSelector
            Fitted selector.
        """
        X, y = self._validate(X, y)

        if self.task == 'classification':
            scores, pvalues = f_classif(X, y)
        elif self.task == 'regression':
            scores, pvalues = f_regression(X, y)
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        # Constant features give NaN; rank them last.
        self.scores_ = np.nan_to_num(scores, nan=0.0)
        self.pvalues_ = np.nan_to_num(pvalues, nan=1.0)
        self.feature_importances_ = self.scores_

        if self.n_features is not None:
            self.support_ = self._top(self.scores_)
        else:
            self.support_ = self.pvalues_ < self.alpha
        return self
