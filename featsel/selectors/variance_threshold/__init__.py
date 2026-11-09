"""
Variance threshold: drop features whose variance is below a threshold.
"""

import numpy as np
from sklearn.feature_selection import VarianceThreshold as SKLearnVarThreshold

from .. import gpu
from ..base import FeatureSelector

# Peak bytes per value of a GPU batch: the float64 copy and its centered copy.
_BYTES_PER_VALUE = 16


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

    device : str, default='auto'
        Where the variances are computed: 'auto' uses a CUDA GPU when one is
        free (the `gpu` extra), else NumPy; 'cpu' always scikit-learn's own
        VarianceThreshold; 'cuda' always the GPU, waiting while another fit
        holds it. A tensor on a GPU runs there whatever this says. The GPU
        sums in float64 and rounds differently, so a feature whose variance
        is at the threshold to the last digits can fall on the other side.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    device_ : str
        The device the variances were computed on: 'cpu' or 'cuda'.
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

    def __init__(self, threshold=0.0, device='auto', check_input=True):
        self.threshold = threshold
        self.device = device
        self.check_input = check_input

    def _fit(self, X, y=None):
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
        X, _ = self._validate(X, requires_y=False, keep_gpu_tensor=True)

        self.variances_, self.device_ = gpu.run(
            self.device, X,
            lambda X: SKLearnVarThreshold(threshold=self.threshold).fit(X).variances_,
            self._variances_torch)
        if np.all(~np.isfinite(self.variances_) | (self.variances_ <= self.threshold)):
            # scikit-learn's own message; on the CPU it has raised it already.
            raise ValueError(f"No feature in X meets the variance threshold {self.threshold:.5f}")
        self.feature_importances_ = self.variances_
        self.support_ = self.variances_ > self.threshold
        return self

    def _variances_torch(self, X, device):
        """scikit-learn's variances, computed on device in float64."""
        torch = gpu.import_torch(device)
        parts = []
        for block in gpu.float64_columns(torch, X, device, _BYTES_PER_VALUE):
            variances = block.var(0, correction=0)
            if self.threshold == 0:
                # As scikit-learn: a constant feature's rounded variance can be
                # above 0, its range cannot.
                variances = torch.minimum(variances, block.amax(0) - block.amin(0))
            parts.append(variances.cpu().numpy())
        return np.concatenate(parts)
