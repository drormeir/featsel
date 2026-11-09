"""
Recursive feature elimination: drop the weakest features of a linear SVM, refit, repeat.
"""

import numpy as np
from sklearn.svm import LinearSVC

from .. import gpu
from ..base import FeatureSelectorRandom
from . import linear_svm


class RFESelector(FeatureSelectorRandom):
    """
    Select features by recursive feature elimination with a linear SVM (SVM-RFE).

    Fits a linear SVM, removes the features with the smallest squared
    weights, and refits on the rest until n_features remain. This is the
    gene-selection method of Guyon et al. (2002). Classification only.

    Selects the same features as scikit-learn's
    RFE(LinearSVC(C=C), step=step), and also ranks every feature.

    ranking() orders every feature: by the round it was removed in, later
    first, and within a round by its weight then. With a fixed step,
    the top k of that ranking is RFE's selection for any k >= n_features, so
    one fit to the smallest k serves every larger k.

    With solver='newton' each round's SVM is solved exactly, by Newton's
    method (see linear_svm.py), instead of by liblinear to a tolerance. Its
    selection is liblinear's at a tight tolerance. At liblinear's default
    tolerance, 1e-4, the two can differ where features' weights nearly tie,
    as correlated genes' weights do: the default stops before those ties
    are resolved.

    Best for: multivariate selection tailored to a linear classifier
    Pros: judges each feature alongside the others, not alone
    Cons: one SVM fit per round, so slow on thousands of features; the
    ranking depends on the SVM's regularization

    Parameters
    ----------
    n_features : int
        Number of features to keep. Required at fit time.
    step : int or float, default=0.1
        Features removed per round. A float in (0, 1) is a fraction of the
        starting feature count, so 0.1 takes about ten rounds. scikit-learn's
        default of 1 refits once per feature, which is impractical here.
    C : float, default=1.0
        Inverse regularization strength of the SVM.
    max_iter : int, default=1000
        Maximum solver iterations per SVM fit.
    random_state : int, default=42
        Random seed for the SVM solver. The GPU solver is deterministic.
    solver : {'liblinear', 'newton'}, default='liblinear'
        'liblinear' fits scikit-learn's LinearSVC. 'newton' solves the same
        SVM exactly, in float64.
    device : str, default='cpu'
        Where the newton solver's products with X run: 'cpu' in NumPy;
        'cuda', or another PyTorch device, there, waiting while another fit
        holds the GPU; 'auto' on a GPU when one is present. A tensor X on a
        GPU runs there. liblinear runs on the CPU only.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    device_ : str
        The device the SVM fits ran on.
    ranking_ : np.ndarray
        1 for kept features; larger numbers were removed in earlier rounds.
    elimination_order_ : np.ndarray
        Every feature index in the order removed, first removed first; the
        survivors last, weakest first.

    Examples
    --------
    >>> from featsel.selectors import RFESelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> RFESelector(n_features=5, random_state=42).fit_transform(X, y).shape
    (100, 5)
    """

    rank_once = True

    def __init__(self, n_features=None, step=0.1, C=1.0, max_iter=1000,
                 random_state=42, solver='liblinear', device='cpu', check_input=True):
        super().__init__(random_state)
        self.n_features = n_features
        self.step = step
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.device = device
        self.check_input = check_input

    def _fit(self, X, y):
        """
        Eliminate features until n_features remain.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : RFESelector
            Fitted selector.
        """
        if self.n_features is None:
            raise ValueError("RFESelector requires n_features to be specified")
        if self.solver == 'liblinear':
            if self.device != 'cpu':
                raise ValueError("solver='liblinear' runs on the CPU only; "
                                 "use solver='newton' for another device")
            X, y = self._validate(X, y, check_input=self.check_input)
            self.device_ = 'cpu'
            self._eliminate(X.shape[1],
                            lambda features: self._liblinear_weights(X[:, features], y))
            return self
        if self.solver != 'newton':
            raise ValueError(f"solver must be 'liblinear' or 'newton', got {self.solver!r}")

        X, y = self._validate(X, y, keep_gpu_tensor=True, check_input=self.check_input)
        self.device_ = gpu.resolve(self.device, X)
        with gpu.exclusive(self.device_):
            rounds = linear_svm.Elimination(self._newton_input(X), y, np.unique(y),
                                            self.C, self.max_iter)
            self._eliminate(X.shape[1], rounds.squared_weights)
        return self

    def _newton_input(self, X):
        """X in float64 on device_: a NumPy array for the CPU, else a tensor."""
        if self.device_ == 'cpu':
            return np.asarray(X, dtype=np.float64)
        torch = gpu.import_torch(self.device_)
        return torch.as_tensor(X if isinstance(X, torch.Tensor)
                               else np.require(X, requirements=['W']),
                               dtype=torch.float64, device=self.device_)

    def _liblinear_weights(self, X, y):
        """Squared LinearSVC weights per feature, summed over classes, as RFE ranks."""
        svm = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        svm.fit(X, y)
        return (svm.coef_ ** 2).sum(axis=0)

    def _eliminate(self, n_total, squared_weights, after_fit=None):
        """
        Remove the features with the smallest squared weights, round by round.

        squared_weights(features) gives the weights of the surviving features,
        sorted indices, from an SVM fit on them alone. after_fit(features),
        if given, is called after every fit, including one more fit on the
        final survivors, as scikit-learn's RFE scores its steps for RFECV.
        """
        # As scikit-learn's RFE: a fraction of the starting count, at least 1.
        step = int(max(1, self.step * n_total)) if 0 < self.step < 1 else int(self.step)
        support = np.ones(n_total, dtype=bool)
        ranking = np.ones(n_total, dtype=int)
        removed = []               # each round's removed features, weakest first
        kept = np.arange(n_total)  # the survivors, weakest first

        while support.sum() > self.n_features:
            features = np.flatnonzero(support)
            # As scikit-learn's RFE: squared weights, summed over classes.
            order = np.argsort(squared_weights(features), kind='stable')
            if after_fit is not None:
                after_fit(features)
            cut = min(step, len(features) - self.n_features)
            dropped, kept = features[order[:cut]], features[order[cut:]]
            removed.append(dropped)
            support[dropped] = False
            ranking[~support] += 1
        if after_fit is not None:
            features = np.flatnonzero(support)
            squared_weights(features)
            after_fit(features)

        self.elimination_order_ = np.concatenate([*removed, kept])
        self.ranking_ = ranking
        self.support_ = support
        return self

    def ranking(self):
        """Feature indices kept longest first: elimination_order_ reversed."""
        return self.elimination_order_[::-1]
