"""
Lasso: rank features by the size of L1-regularized model coefficients.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .. import gpu
from ..base import FeatureSelector
from ..rfe.linear_svm import to_numpy
from . import ppdna


class LassoSelector(FeatureSelector):
    """
    Select features by the magnitude of L1-regularized model coefficients.

    Best for: sparse linear signal, correlated features that should be pruned
    Pros: multivariate, ranks and sparsifies in one fit, well understood theory
    Cons: costs a model fit, unstable when features are strongly correlated,
    picks roughly one feature per correlated group

    This is L1-penalized logistic regression, scikit-learn's
    LogisticRegression(l1_ratio=1) objective. Classification only. Feature
    importance is the absolute coefficient, summed over classes in the
    multiclass case.

    The model is solved by a dual Newton proximal point method, PPDNA (see
    ppdna.py), to a tight tolerance, on a GPU or the CPU. Its selection is
    liblinear's at a tight tolerance. At liblinear's default tolerance,
    1e-4, liblinear stops early and keeps different features.

    C sets how many coefficients are non-zero. Given Cs, a list of values,
    the selector chooses C by inner stratified cross-validation: for each
    value, it fits the model on each fold's training part and scores the
    model's ROC AUC on the fold's held-out part. The value with the best
    mean score wins; a tie goes to the smaller C, which keeps fewer
    features. The model is then fit with that C on all of X. This costs
    cv * len(Cs) + 1 fits.

    Parameters
    ----------
    n_features : int, optional
        At most this many features, the largest absolute coefficients. Only
        features with a non-zero coefficient are selected, so fewer are
        selected when fewer are non-zero. If None, all features with a
        non-zero coefficient are selected, which lets the method choose its
        own feature count.
    C : float, default=1.0
        Inverse regularization strength. Smaller is sparser. Ignored when
        Cs is given.
    Cs : list of float, optional
        Values of C to choose from by cross-validation. Two classes only.
    cv : int, default=5
        Number of inner stratified folds, with Cs. Not shuffled, so a fit
        repeats.
    max_iter : int, default=1000
        Maximum PPDNA outer steps.
    device : str, default='cpu'
        Where PPDNA runs: 'cpu'; 'cuda', or another PyTorch device, there,
        waiting while another fit holds the GPU; 'auto' on a GPU when one is
        present. A tensor X on a GPU runs there.

    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    device_ : str
        The device the fit ran on.
    coef_ : np.ndarray
        Fitted model coefficients.
    C_ : float
        The C of the final fit: C, or the value chosen from Cs.
    cv_results_ : dict
        With Cs only: 'C', the values sorted, and per value the mean, the
        standard deviation and each fold's validation ROC AUC.

    Examples
    --------
    >>> from featsel.selectors import LassoSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> LassoSelector(n_features=10).fit_transform(X, y).shape
    (100, 10)
    """

    rank_once = True

    def __init__(self, n_features=None, C=1.0, Cs=None, cv=5, max_iter=1000, device='cpu',
                 check_input=True):
        self.n_features = n_features
        self.C = C
        self.Cs = Cs
        self.cv = cv
        self.max_iter = max_iter
        self.device = device
        self.check_input = check_input

    def _fit(self, X, y):
        """
        Fit the L1 model and rank features by absolute coefficient.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : LassoSelector
            Fitted selector.
        """
        X, y = self._validate(X, y, keep_gpu_tensor=True, check_input=self.check_input)
        self.device_ = gpu.resolve(self.device, X)
        classes = np.unique(y)
        with gpu.exclusive(self.device_):
            self.C_ = self.C if self.Cs is None else self._best_C(X, y, classes)
            self.coef_ = self._coef(X, y, classes, self.C_)[0]

        self.support_ = self._mask(self.ranking()[:self.n_features])
        if not self.support_.any():
            raise ValueError("L1 regularization zeroed every coefficient. Increase C.")
        return self

    def searches(self):
        """Whether fit chooses C from Cs."""
        return self.Cs is not None

    def fixed(self):
        """This selector with the chosen C and no search."""
        return clone(self).set_params(C=self.C_, Cs=None)

    def _coef(self, X, y, classes, C):
        """
        Coefficients, one row per problem, and intercepts. As liblinear: the
        second class against the first, or each class against the rest.
        """
        fits = [ppdna.l1_logistic(X, (y == label) * 2.0 - 1.0, C, self.device_, self.max_iter)
                for label in (classes[1:] if len(classes) == 2 else classes)]
        return np.array([coef for coef, _ in fits]), np.array([b for _, b in fits])

    def _best_C(self, X, y, classes):
        """The value in Cs with the best mean validation ROC AUC; a tie goes to the smaller."""
        if len(classes) != 2:
            raise ValueError(f"Cs needs two classes, got {len(classes)}")
        Cs = np.sort(np.asarray(self.Cs, dtype=float))
        folds = StratifiedKFold(self.cv).split(np.zeros((len(y), 1)), y)
        scores = []
        for train, test in folds:
            X_train, X_test = X[train], to_numpy(X[test])
            positive = y[test] == classes[1]
            scores.append([])
            for C in Cs:
                coef, intercept = self._coef(X_train, y[train], classes, C)
                scores[-1].append(roc_auc_score(positive, X_test @ coef[0] + intercept[0]))
        scores = np.array(scores)
        mean = scores.mean(axis=0)
        self.cv_results_ = {
            'C': Cs, 'mean_test_score': mean, 'std_test_score': scores.std(axis=0),
            **{f'split{i}_test_score': fold for i, fold in enumerate(scores)},
        }
        return float(Cs[np.argmax(mean)])   # argmax takes the first best: the smaller C

    def ranking(self):
        """
        Features with a non-zero coefficient, largest absolute value first.

        A zero coefficient means the model dropped the feature, so it is never
        selected, even when fewer than n_features are non-zero.
        """
        # Multiclass gives one coefficient row per class; a feature matters if
        # it matters for any class, so sum the absolute values.
        scores = np.abs(np.atleast_2d(self.coef_)).sum(axis=0)
        ranking = self._best_first(scores)
        return ranking[scores[ranking] != 0]
