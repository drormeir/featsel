"""
Sequential selection: add (or remove) one feature at a time by cross-validated score.
"""

import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import check_cv

from .. import gpu
from ..base import FeatureSelector
from . import logistic


class SequentialSelector(FeatureSelector):
    """
    Greedy forward or backward selection with logistic regression.

    Forward: start from no features and, at each step, add the one feature
    whose addition gives the best cross-validated score. Backward: start from
    all and remove one at a time. Stops at n_features. Classification only.

    With solver='lbfgs' it wraps scikit-learn's SequentialFeatureSelector.
    With solver='newton' it runs the same forward procedure, but fits all of
    a step's candidates in a fold at once, by batched Newton's method in
    PyTorch (see logistic.py), on the CPU or a GPU. It selects what
    scikit-learn does with a converged LogisticRegression. It needs a
    binary target, direction='forward' and accuracy as the score.

    Best for: small feature sets, or a short list after a fast filter
    Pros: judges features by the classifier's held-out performance
    Cons: one cross-validation per candidate feature per step, so forward
    selection of k from p features costs about k * p * cv model fits;
    impractical on thousands of features without a prefilter

    Parameters
    ----------
    n_features : int
        Number of features to keep. Required at fit time.
    direction : str, default='forward'
        'forward' adds features, 'backward' removes them.
    cv : int, default=5
        Number of inner stratified folds per candidate.
    scoring : str, optional
        scikit-learn scoring name, such as 'f1_macro'. None uses accuracy.
    C : float, default=1.0
        Inverse regularization strength of the logistic regression.
    max_iter : int, default=1000
        Maximum solver iterations per fit.
    n_jobs : int, default=1
        Candidates evaluated in parallel, with solver='lbfgs'.
    solver : {'lbfgs', 'newton'}, default='lbfgs'
        'lbfgs' fits each candidate with scikit-learn's LogisticRegression.
        'newton' solves every candidate of a fold together, exactly; it
        needs PyTorch (the `gpu` extra), even on the CPU.
    device : str, default='cpu'
        Where the newton solver runs: 'cpu', 'cuda' or another PyTorch
        device, or 'auto' for a GPU when one is present. A tensor X on a
        GPU runs there.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    selection_order_ : list of int
        The chosen features in the order they were added; newton only.
    device_ : str
        The device the fits ran on.

    Examples
    --------
    >>> from featsel.selectors import SequentialSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=10, n_informative=3)
    >>> SequentialSelector(n_features=3, cv=3).fit_transform(X, y).shape
    (100, 3)
    """

    def __init__(self, n_features=None, direction='forward', cv=5, scoring=None,
                 C=1.0, max_iter=1000, n_jobs=1, solver='lbfgs', device='cpu',
                 check_input=True):
        self.n_features = n_features
        self.direction = direction
        self.cv = cv
        self.scoring = scoring
        self.C = C
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.solver = solver
        self.device = device
        self.check_input = check_input

    def fit(self, X, y):
        """
        Add or remove features one at a time until n_features remain.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : SequentialSelector
            Fitted selector.
        """
        self._require_n_features()
        if self.solver == 'newton':
            return self._fit_newton(X, y)
        if self.solver != 'lbfgs':
            raise ValueError(f"solver must be 'lbfgs' or 'newton', got {self.solver!r}")
        if self.device != 'cpu':
            raise ValueError("solver='lbfgs' runs on the CPU only; "
                             "use solver='newton' for another device")
        X, y = self._validate(X, y)
        self.device_ = 'cpu'

        model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        sfs = SequentialFeatureSelector(
            model, n_features_to_select=self.n_features, direction=self.direction,
            scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs
        ).fit(X, y)

        self.support_ = sfs.support_
        return self

    def _fit_newton(self, X, y):
        """Forward selection with each step's candidates solved together."""
        if self.direction != 'forward':
            raise ValueError("solver='newton' supports direction='forward' only")
        if self.scoring not in (None, 'accuracy'):
            raise ValueError("solver='newton' scores by accuracy only")
        X, y = self._validate(X, y, keep_gpu_tensor=True)
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError(f"solver='newton' needs two classes, got {len(labels)}")
        self.device_ = gpu.resolve(self.device, X)
        folds = list(check_cv(self.cv, y, classifier=True).split(np.zeros((len(y), 1)), y))
        torch = gpu.import_torch(self.device_)
        with gpu.exclusive(self.device_):
            X = torch.as_tensor(X if isinstance(X, torch.Tensor)
                                else np.require(X, requirements=['W']),
                                dtype=torch.float64, device=self.device_)
            self.selection_order_ = logistic.forward_selection(
                torch, X, (y == labels[1]).astype(float), self.n_features, folds,
                self.C, self.max_iter, gpu)
        self.support_ = self._mask(self.selection_order_)
        return self
