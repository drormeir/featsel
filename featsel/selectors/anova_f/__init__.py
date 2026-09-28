"""
ANOVA F-test: rank features by their univariate F statistic.
"""

import numpy as np
from sklearn.feature_selection import f_classif, f_regression

from ..base import BaseSelector


class ANOVAFSelector(BaseSelector):
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
    **kwargs : dict
        Additional arguments (unused).

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
    >>> selector = ANOVAFSelector(n_features=10)
    >>> selector.fit(X, y)
    >>> X_selected = selector.transform(X)
    >>> X_selected.shape
    (100, 10)
    """

    def __init__(self, n_features=None, alpha=0.05, task='classification', **kwargs):
        super().__init__(n_features=n_features, **kwargs)
        self.alpha = alpha
        self.task = task

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
        if y is None:
            raise ValueError("ANOVAFSelector requires target values (y)")

        self._store_feature_info(X)
        X_array = self._convert_to_array(X)
        y_array = self._convert_to_series(y)

        # Compute F-statistics and p-values
        if self.task == 'classification':
            self.scores_, self.pvalues_ = f_classif(X_array, y_array)
        elif self.task == 'regression':
            self.scores_, self.pvalues_ = f_regression(X_array, y_array)
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        # Handle NaN values in scores (can occur with constant features)
        self.scores_ = np.nan_to_num(self.scores_, nan=0.0)
        self.pvalues_ = np.nan_to_num(self.pvalues_, nan=1.0)

        # Select features
        if self.n_features is not None:
            # Select top n_features by F-score
            self.selected_indices_ = np.argsort(self.scores_)[-self.n_features:]
        else:
            # Select by p-value threshold
            self.selected_indices_ = np.where(self.pvalues_ < self.alpha)[0]

        self.feature_importances_ = self.scores_
        self.is_fitted_ = True
        return self

    def get_support(self, indices=False):
        """
        Get boolean mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return integer indices.
            If False, return boolean mask.

        Returns
        -------
        support : np.ndarray
            Boolean mask or integer indices of selected features.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before get_support")

        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_indices_] = True
        return self.selected_indices_ if indices else support
