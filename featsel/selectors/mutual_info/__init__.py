"""
Mutual information: rank features by their dependence on the target.
"""

import sys

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .. import gpu
from ..base import FeatureSelectorRandom
from . import knn_1d


class MutualInfoSelector(FeatureSelectorRandom):
    """
    Select features based on mutual information with target.

    Best for: Capturing non-linear relationships, both classification and regression
    Pros: Detects any relationship (linear/non-linear), model-agnostic
    Cons: Computationally expensive (O(n*m*log(n))), requires hyperparameter tuning

    Classification computes scikit-learn's mutual_info_classif by sorting
    each feature instead of building KD-trees (see knn_1d.py), with the same
    result, bit for bit. A class of 2 to 7 samples, where scikit-learn
    switches to brute force, goes to scikit-learn itself, as does regression.

    The sorted estimate can run on a GPU, with the same result, bit for
    bit: X stays there, and only scikit-learn's random noise is drawn on the
    CPU and sent over.

    Parameters
    ----------
    n_features : int
        Number of top features to select. Required at fit time.
    task : str, default='classification'
        Type of task: 'classification' or 'regression'.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
        Lower values = more sensitive to local structure.
    random_state : int, default=42
        Random seed for reproducibility.
    device : str, default='auto'
        Where the sorted estimate runs: 'auto' uses a CUDA GPU when one can
        be used (the `gpu` extra), else NumPy; 'cpu' always NumPy; 'cuda'
        always the GPU, waiting while another fit holds it and raising if it
        cannot run. A tensor X on a GPU runs there. Regression and a class of
        2 to 7 samples run in scikit-learn on the CPU, whatever the device.
        The result is identical on every device.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    device_ : str
        The device the estimate ran on: 'cpu' or 'cuda'.
    scores_ : np.ndarray
        Mutual information of each feature with the target.

    Examples
    --------
    >>> from featsel.selectors import MutualInfoSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> MutualInfoSelector(n_features=10, random_state=42).fit_transform(X, y).shape
    (100, 10)
    """

    def __init__(self, n_features=None, task='classification', n_neighbors=3,
                 random_state=42, device='auto', check_input=True):
        super().__init__(random_state)
        self.n_features = n_features
        self.task = task
        self.n_neighbors = n_neighbors
        self.device = device
        self.check_input = check_input

    def _fit(self, X, y):
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
        if self.n_features is None:
            raise ValueError("MutualInfoSelector requires n_features to be specified")
        X, y = self._validate(X, y, keep_gpu_tensor=True, check_input=self.check_input)

        if self.task == 'classification' and knn_1d.matches_sklearn(y, self.n_neighbors):
            # The same estimate by sorting; scikit-learn's result exactly.
            def mi(X, device='cpu'):
                return knn_1d.mutual_info_classif(X, y, self.n_neighbors,
                                                  self.random_state, device)
            self.scores_, self.device_ = gpu.run(self.device, X, mi, mi)
        else:
            mi_func = self._by_task(self.task, mutual_info_classif, mutual_info_regression)
            self.scores_ = mi_func(
                _on_cpu(X), y, n_neighbors=self.n_neighbors, random_state=self.random_state
            )
            self.device_ = 'cpu'
        self.support_ = self._mask(self.ranking()[:self.n_features])
        return self

    def ranking(self):
        """Feature indices by mutual information, largest first."""
        return self._best_first(self.scores_)


def _on_cpu(X):
    """X, a GPU tensor copied to a NumPy array; anything else as it is."""
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(X, torch.Tensor):
        return X.cpu().numpy()
    return X
