"""
Baseline feature selection methods.

Baselines ignore the data and exist to make the other methods interpretable:
a selector that does not beat random selection contributed nothing.
"""

import numpy as np

from .base import BaseSelector


class RandomSelector(BaseSelector):
    """
    Select n_features columns uniformly at random, ignoring X values and y.

    Best for: the control condition in a method comparison
    Pros: trivially fast, unbiased reference point
    Cons: no relationship with the target, by construction

    Parameters
    ----------
    n_features : int
        Number of features to select. If larger than the number of available
        features, all features are selected.
    random_state : int, optional
        Random seed for reproducibility. Selection is redrawn on every fit(),
        so a fixed seed is what makes cross-validation folds reproducible.
    **kwargs : dict
        Additional arguments (unused, for API consistency).

    Attributes
    ----------
    selected_indices_ : np.ndarray
        Integer indices of the randomly selected features.

    Examples
    --------
    >>> from featsel.selectors.baseline import RandomSelector
    >>> import numpy as np
    >>> X = np.random.randn(100, 20)
    >>> selector = RandomSelector(n_features=5, random_state=42)
    >>> selector.fit(X)
    >>> selector.transform(X).shape
    (100, 5)
    """

    def __init__(self, n_features, random_state=None, **kwargs):
        super().__init__(n_features=n_features, **kwargs)
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Draw a random subset of feature indices.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data. Only its shape is used.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : RandomSelector
            Fitted selector.
        """
        if self.n_features is None:
            raise ValueError("RandomSelector requires n_features to be specified")
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}")

        self._store_feature_info(X)

        rng = np.random.default_rng(self.random_state)
        n_select = min(self.n_features, self.n_features_in_)
        self.selected_indices_ = np.sort(
            rng.choice(self.n_features_in_, size=n_select, replace=False)
        )

        # No notion of importance: every feature is equally (un)informative.
        self.feature_importances_ = np.zeros(self.n_features_in_)
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
