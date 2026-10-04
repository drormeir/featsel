"""
Recursive feature elimination that picks its own feature count by cross-validation.
"""

from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC

from ..base import FeatureSelector


class RFECVSelector(FeatureSelector):
    """
    SVM-RFE with the number of features chosen by cross-validation.

    Runs recursive feature elimination with a linear SVM (see RFESelector)
    inside each fold of an inner stratified cross-validation, scores every
    feature count it passes, and keeps the count with the best mean score.
    Classification only.

    Wraps scikit-learn's RFECV for now.

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
    random_state : int, optional
        Random seed for the SVM solver.
    n_jobs : int, default=1
        Folds evaluated in parallel.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    n_features_ : int
        The feature count chosen by cross-validation.
    ranking_ : np.ndarray
        1 for kept features; larger numbers were removed in earlier rounds.
    cv_results_ : dict
        Mean and per-fold scores for each feature count tried.

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
                 C=1.0, max_iter=1000, random_state=None, n_jobs=1, check_input=True):
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
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
        X, y = self._validate(X, y)

        svm = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        rfecv = RFECV(svm, step=self.step, cv=self.cv, scoring=self.scoring,
                      min_features_to_select=self.min_features_to_select,
                      n_jobs=self.n_jobs).fit(X, y)

        self.n_features_ = rfecv.n_features_
        self.ranking_ = rfecv.ranking_
        self.cv_results_ = rfecv.cv_results_
        self.support_ = rfecv.support_
        return self
