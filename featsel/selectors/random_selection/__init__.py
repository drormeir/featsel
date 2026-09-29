"""
Random selection: the control every other selector must beat.

The folder is not named `random`, which would shadow the standard library
module whenever it lands on the import path.
"""

import numpy as np

from ..base import FeatureSelector


class RandomSelector(FeatureSelector):
    """
    Select n_features columns uniformly at random, ignoring X values and y.

    Best for: the control condition in a method comparison
    Pros: trivially fast, unbiased reference point
    Cons: no relationship with the target, by construction

    Parameters
    ----------
    n_features : int
        Number of features to select. Required at fit time. If larger than the
        number of available features, all features are selected.
    random_state : int, optional
        Random seed for reproducibility. Selection is redrawn on every fit(),
        so a fixed seed is what makes resampling splits reproducible.

    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    feature_importances_ : np.ndarray
        All zeros: no feature is more informative than another.

    Examples
    --------
    >>> from featsel.selectors import RandomSelector
    >>> import numpy as np
    >>> X = np.random.randn(100, 20)
    >>> RandomSelector(n_features=5, random_state=42).fit_transform(X).shape
    (100, 5)
    """

    def __init__(self, n_features=None, random_state=None, check_input=True):
        self.n_features = n_features
        self.random_state = random_state
        self.check_input = check_input

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
        self._require_n_features()
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}")
        self._validate(X, requires_y=False)

        rng = np.random.default_rng(self.random_state)
        n_select = min(self.n_features, self.n_features_in_)
        self.support_ = self._mask(rng.choice(self.n_features_in_, size=n_select, replace=False))

        # No notion of importance: every feature is equally (un)informative.
        self.feature_importances_ = np.zeros(self.n_features_in_)
        return self
