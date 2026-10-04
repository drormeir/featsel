"""
Sequential selection: add (or remove) one feature at a time by cross-validated score.
"""

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

from ..base import FeatureSelector


class SequentialSelector(FeatureSelector):
    """
    Greedy forward or backward selection with logistic regression.

    Forward: start from no features and, at each step, add the one feature
    whose addition gives the best cross-validated score. Backward: start from
    all and remove one at a time. Stops at n_features. Classification only.

    Wraps scikit-learn's SequentialFeatureSelector.

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
        Candidates evaluated in parallel.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Examples
    --------
    >>> from featsel.selectors import SequentialSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=10, n_informative=3)
    >>> SequentialSelector(n_features=3, cv=3).fit_transform(X, y).shape
    (100, 3)
    """

    def __init__(self, n_features=None, direction='forward', cv=5, scoring=None,
                 C=1.0, max_iter=1000, n_jobs=1, check_input=True):
        self.n_features = n_features
        self.direction = direction
        self.cv = cv
        self.scoring = scoring
        self.C = C
        self.max_iter = max_iter
        self.n_jobs = n_jobs
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
        X, y = self._validate(X, y)

        model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        sfs = SequentialFeatureSelector(
            model, n_features_to_select=self.n_features, direction=self.direction,
            scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs
        ).fit(X, y)

        self.support_ = sfs.support_
        return self
