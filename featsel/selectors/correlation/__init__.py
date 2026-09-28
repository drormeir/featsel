"""
Correlation: rank features by target correlation and drop redundant pairs.
"""

import numpy as np
import pandas as pd

from ..base import BaseSelector


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
    >>> from featsel.selectors import CorrelationSelector
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
