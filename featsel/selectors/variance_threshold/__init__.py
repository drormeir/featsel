"""
Variance threshold: drop features whose variance is below a threshold.
"""

from sklearn.feature_selection import VarianceThreshold as SKLearnVarThreshold

from ..base import FeatureSelector


class VarianceThreshold(FeatureSelector):
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

    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    variances_ : np.ndarray
        Variance of each feature.
    feature_importances_ : np.ndarray
        Alias for variances_.

    Examples
    --------
    >>> from featsel.selectors import VarianceThreshold
    >>> import numpy as np
    >>> X = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]])
    >>> VarianceThreshold(threshold=0.1).fit_transform(X).shape
    (4, 2)
    """

    def __init__(self, threshold=0.0, check_input=True):
        self.threshold = threshold
        self.check_input = check_input

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
        X, _ = self._validate(X, requires_y=False)

        selector = SKLearnVarThreshold(threshold=self.threshold).fit(X)
        self.variances_ = selector.variances_
        self.feature_importances_ = self.variances_
        self.support_ = selector.get_support()
        return self
