"""
Filter-based feature selection methods.

Filter methods are fast, model-agnostic techniques that rank features based on
statistical properties and their relationship with the target variable.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold as SKLearnVarThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression
)

from .base import BaseSelector


class VarianceThreshold(BaseSelector):
    """
    Remove features with variance below threshold.

    Best for: Quick preprocessing, removing constant/near-constant features
    Pros: Very fast (O(n*m)), unsupervised
    Cons: Ignores relationship with target variable

    Parameters
    ----------
    threshold : float, default=0.0
        Features with variance below this threshold will be removed.
        Default (0.0) removes features with zero variance.
    **kwargs : dict
        Additional arguments (unused, for API consistency).

    Attributes
    ----------
    variances_ : np.ndarray
        Variance of each feature.

    Examples
    --------
    >>> from featsel.selectors.filter import VarianceThreshold
    >>> import numpy as np
    >>> X = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]])
    >>> selector = VarianceThreshold(threshold=0.1)
    >>> selector.fit(X)
    >>> X_selected = selector.transform(X)
    >>> X_selected.shape
    (4, 2)
    """

    def __init__(self, threshold=0.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self._selector = SKLearnVarThreshold(threshold=threshold)

    def fit(self, X, y=None):
        """
        Fit the selector on training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : VarianceThreshold
            Fitted selector.
        """
        self._store_feature_info(X)
        X_array = self._convert_to_array(X)

        self._selector.fit(X_array)
        self.variances_ = self._selector.variances_
        self.feature_importances_ = self.variances_
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
        return self._selector.get_support(indices=indices)


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
    >>> from featsel.selectors.filter import ANOVAFSelector
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
    >>> from featsel.selectors.filter import MutualInfoSelector
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


class CorrelationSelector(BaseSelector):
    """
    Select features based on correlation with target and remove redundant features.

    Best for: Removing redundancy, simple linear relationships
    Pros: Simple (O(m^2*n)), interpretable, reduces multicollinearity
    Cons: Only linear correlation, may remove important correlated features

    Strategy:
    1. Compute correlation of each feature with target
    2. Keep features above target_threshold
    3. Among highly correlated feature pairs, keep the one with higher target correlation
    4. Select top n_features by target correlation (if specified)

    Parameters
    ----------
    n_features : int, optional
        Number of features to select. If None, uses target_threshold only.
    target_threshold : float, default=0.1
        Minimum absolute correlation with target to keep feature.
    inter_feature_threshold : float, default=0.95
        Maximum correlation between features (remove one if exceeded).
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'.
    **kwargs : dict
        Additional arguments (unused).

    Attributes
    ----------
    target_corr_ : pd.Series
        Absolute correlation of each feature with target.
    feature_importances_ : np.ndarray
        Alias for target_corr_ values.
    selected_features_ : list
        Names or indices of selected features.

    Examples
    --------
    >>> from featsel.selectors.filter import CorrelationSelector
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    >>> y = X['f0'] + X['f1'] + np.random.randn(100) * 0.1
    >>> selector = CorrelationSelector(n_features=3)
    >>> selector.fit(X, y)
    >>> X_selected = selector.transform(X)
    >>> X_selected.shape
    (100, 3)
    """

    def __init__(self, n_features=None, target_threshold=0.1,
                 inter_feature_threshold=0.95, method='pearson', **kwargs):
        super().__init__(n_features=n_features, **kwargs)
        self.target_threshold = target_threshold
        self.inter_feature_threshold = inter_feature_threshold
        self.method = method

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
        self : CorrelationSelector
            Fitted selector.
        """
        if y is None:
            raise ValueError("CorrelationSelector requires target values (y)")

        self._store_feature_info(X)

        # Convert to DataFrame for correlation computation
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_in_:
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')

        # 1. Compute correlation with target
        self.target_corr_ = X.corrwith(y, method=self.method).abs()

        # 2. Keep features above target threshold
        candidates = self.target_corr_[self.target_corr_ >= self.target_threshold].index.tolist()

        if len(candidates) == 0:
            raise ValueError(
                f"No features meet target_threshold={self.target_threshold}. "
                f"Max correlation: {self.target_corr_.max():.3f}"
            )

        # 3. Remove highly correlated feature pairs
        corr_matrix = X[candidates].corr(method=self.method).abs()
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

        to_drop = set()
        for i, j in zip(*np.where((corr_matrix.values > self.inter_feature_threshold) & upper_tri)):
            # Drop feature with lower target correlation
            feat_i = corr_matrix.index[i]
            feat_j = corr_matrix.columns[j]
            if self.target_corr_[feat_i] > self.target_corr_[feat_j]:
                to_drop.add(feat_j)
            else:
                to_drop.add(feat_i)

        selected = [f for f in candidates if f not in to_drop]

        if len(selected) == 0:
            raise ValueError("All features were removed due to inter-feature correlation threshold")

        # 4. Select top n_features by target correlation
        if self.n_features is not None and len(selected) > self.n_features:
            selected = self.target_corr_[selected].nlargest(self.n_features).index.tolist()

        self.selected_features_ = selected
        self.feature_importances_ = self.target_corr_.values
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

        if self.feature_names_in_:
            # Create boolean mask based on feature names
            support = np.array([name in self.selected_features_ for name in self.feature_names_in_])
            selected_indices = np.where(support)[0]
        else:
            # Selected features are indices
            support = np.zeros(self.n_features_in_, dtype=bool)
            support[self.selected_features_] = True
            selected_indices = self.selected_features_

        return selected_indices if indices else support
