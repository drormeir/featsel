"""
Variance threshold: drop features whose variance is below a threshold.
"""

from sklearn.feature_selection import VarianceThreshold as SKLearnVarThreshold

from ..base import BaseSelector


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
    >>> from featsel.selectors import VarianceThreshold
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
