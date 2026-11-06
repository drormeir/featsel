"""
Recursive feature elimination: drop the weakest features of a linear SVM, refit, repeat.
"""

import numpy as np
from sklearn.svm import LinearSVC

from .. import gpu
from ..base import FeatureSelector
from . import linear_svm


class RFESelector(FeatureSelector):
    """
    Select features by recursive feature elimination with a linear SVM (SVM-RFE).

    Fits a linear SVM, removes the features with the smallest squared
    weights, and refits on the rest until n_features remain. This is the
    gene-selection method of Guyon et al. (2002). Classification only.

    Selects the same features as scikit-learn's
    RFE(LinearSVC(C=C), step=step), and also ranks every feature.

    feature_importances_ ranks every feature: by the round it was removed in,
    later first, and within a round by its weight then. With a fixed step,
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
    feature_importances_ : np.ndarray
        Position of each feature in the elimination order: larger was kept
        longer.

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
        self.n_features = n_features
        self.step = step
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.device = device
        self.check_input = check_input

    def fit(self, X, y):
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
        self._require_n_features()
        if self.solver == 'liblinear':
            if self.device != 'cpu':
                raise ValueError("solver='liblinear' runs on the CPU only; "
                                 "use solver='newton' for another device")
            X, y = self._validate(X, y)
            self.device_ = 'cpu'
            self._eliminate(X.shape[1],
                            lambda features: self._liblinear_weights(X[:, features], y))
            return self
        if self.solver != 'newton':
            raise ValueError(f"solver must be 'liblinear' or 'newton', got {self.solver!r}")

        X, y = self._validate(X, y, keep_gpu_tensor=True)
        self.device_ = gpu.resolve(self.device, X)
        classes = np.unique(y)
        if self.device_ == 'cpu':
            rounds = linear_svm.Elimination(np.asarray(X, dtype=np.float64), y, classes,
                                            self.C, self.max_iter)
            self._eliminate(X.shape[1], rounds.squared_weights)
            return self
        torch = gpu.import_torch(self.device_)
        with gpu.exclusive(self.device_):
            X = torch.as_tensor(X if isinstance(X, torch.Tensor)
                                else np.require(X, requirements=['W']),
                                dtype=torch.float64, device=self.device_)
            rounds = linear_svm.Elimination(X, y, classes, self.C, self.max_iter)
            self._eliminate(X.shape[1], rounds.squared_weights)
        return self

    def _liblinear_weights(self, X, y):
        """Squared LinearSVC weights per feature, summed over classes, as RFE ranks."""
        svm = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        svm.fit(X, y)
        return (svm.coef_ ** 2).sum(axis=0)

    def _eliminate(self, n_total, squared_weights):
        """
        Remove the features with the smallest squared weights, round by round.

        squared_weights(features) gives the weights of the surviving features,
        sorted indices, from an SVM fit on them alone.
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
            order = np.argsort(squared_weights(features))
            cut = min(step, len(features) - self.n_features)
            dropped, kept = features[order[:cut]], features[order[cut:]]
            removed.append(dropped)
            support[dropped] = False
            ranking[~support] += 1

        elimination = np.concatenate([*removed, kept])
        self.feature_importances_ = np.empty(n_total)
        self.feature_importances_[elimination] = np.arange(n_total)
        self.ranking_ = ranking
        self.support_ = support
        return self
