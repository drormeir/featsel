"""
Mutual information: rank features by their dependence on the target.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from ..base import BaseSelector


class MutualInfoSelector(BaseSelector):
    """
    Select features based on mutual information with target.

    Best for: Capturing non-linear relationships, both classification and regression
    Pros: Detects any relationship (linear/non-linear), model-agnostic
    Cons: Computationally expensive (O(n*m*log(n))), requires hyperparameter tuning

    Parameters
    ----------
    n_features : int
        Number of top features to select.
    task : str, default='classification'
        Type of task: 'classification' or 'regression'.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
        Lower values = more sensitive to local structure.
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs : dict
        Additional arguments (unused).

    Attributes
    ----------
    feature_importances_ : np.ndarray
        Mutual information scores for each feature.

    Examples
    --------
    >>> from featsel.selectors import MutualInfoSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> selector = MutualInfoSelector(n_features=10, random_state=42)
    >>> selector.fit(X, y)
    >>> X_selected = selector.transform(X)
    >>> X_selected.shape
    (100, 10)
    """

    def __init__(self, n_features, task='classification', n_neighbors=3,
                 random_state=None, **kwargs):
        super().__init__(n_features=n_features, **kwargs)
        self.task = task
        self.n_neighbors = n_neighbors
        self.random_state = random_state

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
        self : MutualInfoSelector
            Fitted selector.
        """
        if y is None:
            raise ValueError("MutualInfoSelector requires target values (y)")

        if self.n_features is None:
            raise ValueError("MutualInfoSelector requires n_features to be specified")

        self._store_feature_info(X)
        X_array = self._convert_to_array(X)
        y_array = self._convert_to_series(y)

        # Compute mutual information
        if self.task == 'classification':
            mi_func = mutual_info_classif
        elif self.task == 'regression':
            mi_func = mutual_info_regression
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        self.feature_importances_ = mi_func(
            X_array, y_array,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state
        )

        # Select top n_features
        self.selected_indices_ = np.argsort(self.feature_importances_)[-self.n_features:]
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
