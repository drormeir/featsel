"""
Correlation: rank features by target correlation and drop redundant pairs.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..base import FeatureSelector

# Candidate features compared per block in the redundancy step. A block's
# correlations are _BLOCK x n_candidates float64 values: 80 MB at 10,000.
_BLOCK = 1024


class CorrelationSelector(FeatureSelector):
    """
    Select features based on correlation with target and remove redundant features.

    Best for: Removing redundancy, simple linear relationships
    Pros: Simple (O(m^2*n)), interpretable, reduces multicollinearity
    Cons: Only linear correlation, may remove important correlated features

    The redundancy step never holds the full correlation matrix. For
    Pearson and Spearman it standardizes the candidates once and finds the
    pairs above the threshold block by block, with one matrix product per
    block. Kendall goes through pandas.

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

    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    target_corr_ : pd.Series
        Absolute correlation of each feature with target.
    feature_importances_ : np.ndarray
        Alias for target_corr_ values.
    selected_features_ : list
        Names (DataFrame input) or indices of selected features.

    Examples
    --------
    >>> from featsel.selectors import CorrelationSelector
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    >>> y = X['f0'] + X['f1'] + np.random.randn(100) * 0.1
    >>> CorrelationSelector(n_features=3).fit_transform(X, y).shape
    (100, 3)
    """

    def __init__(self, n_features=None, target_threshold=0.1,
                 inter_feature_threshold=0.95, method='pearson', check_input=True):
        self.n_features = n_features
        self.target_threshold = target_threshold
        self.inter_feature_threshold = inter_feature_threshold
        self.method = method
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
        self : CorrelationSelector
            Fitted selector.
        """
        X, y = self._validate(X, y)
        names = list(getattr(self, 'feature_names_in_', range(self.n_features_in_)))
        X = pd.DataFrame(X, columns=names)
        y = np.asarray(y)
        if not np.issubdtype(y.dtype, np.number):
            # Two class labels code as 0 and 1; more have no numeric order.
            classes, y = np.unique(y, return_inverse=True)
            if len(classes) != 2:
                raise ValueError(
                    f"A non-numeric target needs exactly two classes, got {len(classes)}")

        # 1. Compute correlation with target
        self.target_corr_ = X.corrwith(pd.Series(y), method=self.method).abs()

        # 2. Keep features above target threshold
        candidates = self.target_corr_[self.target_corr_ >= self.target_threshold].index.tolist()

        if len(candidates) == 0:
            raise ValueError(
                f"No features meet target_threshold={self.target_threshold}. "
                f"Max correlation: {self.target_corr_.max():.3f}"
            )

        # 3. Remove highly correlated feature pairs
        to_drop = set()
        for i, j in self._redundant_pairs(X[candidates]):
            # Drop feature with lower target correlation
            feat_i, feat_j = candidates[i], candidates[j]
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
        self.support_ = np.isin(names, selected)
        return self

    def _redundant_pairs(self, X):
        """Pairs (i, j), i < j, of columns of X correlated above the threshold."""
        if self.method == 'kendall':
            corr = X.corr(method='kendall').abs().to_numpy()
            return zip(*np.nonzero(np.triu(corr > self.inter_feature_threshold, k=1)))
        Z = X.to_numpy(dtype=np.float64)
        if self.method == 'spearman':
            Z = rankdata(Z, axis=0)          # average ranks, as pandas ranks
        Z = Z - Z.mean(axis=0)
        Z /= np.linalg.norm(Z, axis=0)
        pairs = []
        for i0 in range(0, Z.shape[1], _BLOCK):
            # Only columns from i0 on: pairs with i < j.
            corr = np.abs(Z[:, i0:i0 + _BLOCK].T @ Z[:, i0:])
            i, j = np.nonzero(np.triu(corr > self.inter_feature_threshold, k=1))
            pairs.extend(zip(i + i0, j + i0))
        return pairs
