"""
Tree importance: rank features by random forest impurity decrease.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..base import BaseSelector


class TreeImportanceSelector(BaseSelector):
    """
    Select features by random forest impurity-based importance.

    Best for: non-linear signal, interactions between features
    Pros: multivariate, captures non-linearity, no scaling needed
    Cons: costs a forest fit, biased towards high-cardinality features,
    splits importance across correlated features

    Parameters
    ----------
    n_features : int
        Number of top features to select.
    n_estimators : int, default=200
        Number of trees in the forest. More trees give a more stable ranking.
    task : str, default='classification'
        Type of task: 'classification' or 'regression'.
    max_depth : int, optional
        Maximum tree depth. None grows trees fully.
    random_state : int, optional
        Random seed for the forest.
    n_jobs : int, default=1
        Threads used to fit the forest.
    **kwargs : dict
        Additional arguments (unused).

    Attributes
    ----------
    feature_importances_ : np.ndarray
        Mean impurity decrease per feature.

    Examples
    --------
    >>> from featsel.selectors import TreeImportanceSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> selector = TreeImportanceSelector(n_features=10, random_state=42)
    >>> selector.fit(X, y)
    >>> selector.transform(X).shape
    (100, 10)
    """

    def __init__(self, n_features, n_estimators=200, task='classification',
                 max_depth=None, random_state=None, n_jobs=1, **kwargs):
        super().__init__(n_features=n_features, **kwargs)
        self.n_estimators = n_estimators
        self.task = task
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Fit the forest and rank features by impurity decrease.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : TreeImportanceSelector
            Fitted selector.
        """
        if y is None:
            raise ValueError("TreeImportanceSelector requires target values (y)")
        if self.n_features is None:
            raise ValueError("TreeImportanceSelector requires n_features to be specified")

        self._store_feature_info(X)
        X_array = self._convert_to_array(X)
        y_array = self._convert_to_series(y)

        if self.task == 'classification':
            forest_class = RandomForestClassifier
        elif self.task == 'regression':
            forest_class = RandomForestRegressor
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        forest = forest_class(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        forest.fit(X_array, y_array)

        self.feature_importances_ = forest.feature_importances_
        ranked = np.argsort(self.feature_importances_)[::-1]
        self.selected_indices_ = np.sort(ranked[:self.n_features])

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
