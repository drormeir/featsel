"""
Random selection: the control every other selector must beat.

The folder is not named `random`, which would shadow the standard library
module whenever it lands on the import path.
"""

from ..base import FeatureSelectorRandom


class RandomSelector(FeatureSelectorRandom):
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
    random_state : int, default=42
        Random seed for reproducibility. Selection is redrawn on every fit(),
        so a fixed seed is what makes resampling splits reproducible.

    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Examples
    --------
    >>> from featsel.selectors import RandomSelector
    >>> import numpy as np
    >>> X = np.random.randn(100, 20)
    >>> RandomSelector(n_features=5, random_state=42).fit_transform(X).shape
    (100, 5)
    """

    def __init__(self, n_features=None, random_state=42, check_input=True):
        super().__init__(random_state)
        assert n_features is None or n_features >= 1, \
            f"n_features must be >= 1, got {n_features}"
        self.n_features = n_features
        self.check_input = check_input

    def _fit(self, X, y=None):
        """
        Draw a random subset of feature indices.

        Parameters
        ----------
        X : pd.DataFrame, np.ndarray or torch.Tensor of shape (n_samples, n_features)
            Training data. Only its shape is used. A tensor on a GPU stays
            there: it is checked there, not copied to the CPU.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : RandomSelector
            Fitted selector.
        """
        if self.n_features is None:
            raise ValueError("RandomSelector requires n_features to be specified")
        self._validate(X, requires_y=False, keep_gpu_tensor=True, check_input=self.check_input)

        rng = self._rng()
        n_select = min(self.n_features, self.n_features_in_)
        self.support_ = self._mask(rng.choice(self.n_features_in_, size=n_select, replace=False))
        return self
