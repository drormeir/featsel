"""
Recursive feature elimination: drop the weakest features of a linear SVM, refit, repeat.
"""

import numpy as np
from sklearn.svm import LinearSVC

from ..base import FeatureSelector


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
    random_state : int, optional
        Random seed for the SVM solver.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
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
                 random_state=None, check_input=True):
        self.n_features = n_features
        self.step = step
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
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
        X, y = self._validate(X, y)

        n_total = X.shape[1]
        # As scikit-learn's RFE: a fraction of the starting count, at least 1.
        step = int(max(1, self.step * n_total)) if 0 < self.step < 1 else int(self.step)
        support = np.ones(n_total, dtype=bool)
        ranking = np.ones(n_total, dtype=int)
        removed = []               # each round's removed features, weakest first
        kept = np.arange(n_total)  # the survivors, weakest first

        while support.sum() > self.n_features:
            features = np.flatnonzero(support)
            svm = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
            svm.fit(X[:, features], y)
            # As scikit-learn's RFE: squared weights, summed over classes.
            order = np.argsort((svm.coef_ ** 2).sum(axis=0))
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
