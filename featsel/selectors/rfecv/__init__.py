"""
Recursive feature elimination that picks its own feature count by cross-validation.
"""

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.svm import LinearSVC

from .. import gpu
from ..base import FeatureSelector
from ..rfe import RFESelector, linear_svm


class RFECVSelector(FeatureSelector):
    """
    SVM-RFE with the number of features chosen by cross-validation.

    Runs recursive feature elimination with a linear SVM (see RFESelector)
    inside each fold of an inner stratified cross-validation, scores every
    feature count it passes, and keeps the count with the best mean score.
    Classification only.

    With solver='liblinear' it wraps scikit-learn's RFECV. With
    solver='newton' it runs the same procedure with RFESelector's exact
    Newton solver: the same folds, the same score at every step, and the
    same choice of count, the smallest among the best mean scores. Its
    choice is scikit-learn's with a converged LinearSVC.

    Best for: when the right feature count is unknown
    Pros: chooses its own count from held-out performance
    Cons: cv times the cost of RFESelector; inside the runner's splits this
    is a second, nested level of cross-validation

    Parameters
    ----------
    step : int or float, default=0.1
        Features removed per round. A float in (0, 1) is a fraction of the
        starting feature count.
    cv : int, default=5
        Number of inner stratified folds.
    scoring : str, optional
        scikit-learn scoring name, such as 'f1_macro'. None uses accuracy.
    min_features_to_select : int, default=1
        Smallest feature count considered.
    C : float, default=1.0
        Inverse regularization strength of the SVM.
    max_iter : int, default=1000
        Maximum solver iterations per SVM fit.
    random_state : int, default=42
        Random seed for the SVM solver.
    n_jobs : int, default=1
        Folds evaluated in parallel, with solver='liblinear'.
    solver : {'liblinear', 'newton'}, default='liblinear'
        How each SVM is solved; see RFESelector.
    device : str, default='cpu'
        Where the newton solver's products with X run; see RFESelector.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    n_features_ : int
        The feature count chosen by cross-validation.
    ranking_ : np.ndarray
        1 for kept features; larger numbers were removed in earlier rounds.
    cv_results_ : dict
        Mean and per-fold scores for each feature count tried. The newton
        solver leaves out scikit-learn's per-fold supports and rankings.
    device_ : str
        The device the SVM products ran on.

    Examples
    --------
    >>> from featsel.selectors import RFECVSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=5)
    >>> selector = RFECVSelector(random_state=42).fit(X, y)
    >>> bool(selector.get_support().sum() == selector.n_features_)
    True
    """

    def __init__(self, step=0.1, cv=5, scoring=None, min_features_to_select=1,
                 C=1.0, max_iter=1000, random_state=42, n_jobs=1, solver='liblinear',
                 device='cpu', check_input=True):
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.solver = solver
        self.device = device
        self.check_input = check_input

    def fit(self, X, y):
        """
        Eliminate features and keep the count with the best cross-validated score.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : RFECVSelector
            Fitted selector.
        """
        if self.solver == 'newton':
            return self._fit_newton(X, y)
        if self.solver != 'liblinear':
            raise ValueError(f"solver must be 'liblinear' or 'newton', got {self.solver!r}")
        if self.device != 'cpu':
            raise ValueError("solver='liblinear' runs on the CPU only; "
                             "use solver='newton' for another device")
        X, y = self._validate(X, y)
        self.device_ = 'cpu'

        svm = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        rfecv = RFECV(svm, step=self.step, cv=self.cv, scoring=self.scoring,
                      min_features_to_select=self.min_features_to_select,
                      n_jobs=self.n_jobs).fit(X, y)

        self.n_features_ = rfecv.n_features_
        self.ranking_ = rfecv.ranking_
        self.cv_results_ = rfecv.cv_results_
        self.support_ = rfecv.support_
        return self

    def _fit_newton(self, X, y):
        """RFECV with every SVM solved by Newton's method, as scikit-learn runs it."""
        X, y = self._validate(X, y, keep_gpu_tensor=True)
        self.device_ = gpu.resolve(self.device, X)
        n_total = X.shape[1]
        classes = np.unique(y)
        folds = check_cv(self.cv, y, classifier=True).split(np.zeros((len(y), 1)), y)
        scorer = check_scoring(LinearSVC(), scoring=self.scoring)
        rfe = RFESelector(n_features=min(self.min_features_to_select, n_total),
                          step=self.step, C=self.C, max_iter=self.max_iter,
                          solver='newton', device=self.device_)
        rfe.device_ = self.device_

        with gpu.exclusive(self.device_):
            X = rfe._newton_input(X)
            steps = [self._fold_scores(rfe, X, y, train, test, classes, scorer)
                     for train, test in folds]
            scores = np.array([fold for fold, _ in steps])
            counts = steps[0][1]

            # As scikit-learn: counts from fewest up, and the first best mean
            # score among them, so a tie goes to fewer features.
            scores = scores[:, ::-1]
            counts = np.array(counts)[::-1]
            rfe.n_features = int(counts[np.argmax(scores.sum(axis=0))])
            rfe._eliminate(n_total, linear_svm.Elimination(X, y, classes, self.C,
                                                           self.max_iter).squared_weights)

        self.n_features_ = rfe.n_features
        self.ranking_ = rfe.ranking_
        self.support_ = rfe.support_
        self.cv_results_ = {
            'mean_test_score': scores.mean(axis=0),
            'std_test_score': scores.std(axis=0),
            **{f'split{i}_test_score': fold for i, fold in enumerate(scores)},
            'n_features': counts,
        }
        return self

    def _fold_scores(self, rfe, X, y, train, test, classes, scorer):
        """One fold's score after every SVM fit of the elimination, and the feature counts."""
        rounds = linear_svm.Elimination(X[train], y[train], classes, self.C, self.max_iter)
        X_test = linear_svm.to_numpy(X[test])
        scores, counts = [], []

        def score(features):
            counts.append(len(features))
            scores.append(scorer(rounds.linear_svc(), X_test[:, features], y[test]))

        rfe._eliminate(X.shape[1], rounds.squared_weights, after_fit=score)
        return scores, counts
