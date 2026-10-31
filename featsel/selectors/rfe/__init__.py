"""
Recursive feature elimination: drop the weakest features of a linear SVM, refit, repeat.
"""

import numpy as np
from sklearn.svm import SVC

from ..base import FeatureSelector


class RFESelector(FeatureSelector):
    """
    Select features by recursive feature elimination with a linear SVM (SVM-RFE).

    Fits a linear SVM, removes the features with the smallest squared
    weights, and refits on the rest until n_features remain. This is the
    gene-selection method of Guyon et al. (2002). Two classes only.

    Selects the same features as scikit-learn's
    RFE(SVC(kernel='linear', C=C), step=step), computed faster when samples
    are far fewer than features. The SVM is trained on the samples' Gram
    matrix X X^T, as Guyon et al. did. A round then costs one small SVM fit
    plus updating the Gram matrix for the removed features, instead of
    copying the data into the solver.

    feature_importances_ ranks every feature: by the round it was removed in,
    later first, and within a round by its weight then. With a fixed step,
    the top k of that ranking is RFE's selection for any k >= n_features, so
    one fit to the smallest k serves every larger k.

    Best for: multivariate selection tailored to a linear classifier
    Pros: judges each feature alongside the others, not alone
    Cons: one SVM fit per round; the ranking depends on the SVM's
    regularization; memory and time grow with the square of the sample count

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
    >>> RFESelector(n_features=5).fit_transform(X, y).shape
    (100, 5)
    """

    def __init__(self, n_features=None, step=0.1, C=1.0, check_input=True):
        self.n_features = n_features
        self.step = step
        self.C = C
        self.check_input = check_input

    def fit(self, X, y):
        """
        Eliminate features until n_features remain.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Class labels, exactly two classes.

        Returns
        -------
        self : RFESelector
            Fitted selector.
        """
        self._require_n_features()
        X, y = self._validate(X, y)
        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError(f"RFESelector needs two classes, got {n_classes}")

        n_samples, n_total = X.shape
        # As scikit-learn's RFE: a fraction of the starting count, at least 1.
        step = int(max(1, self.step * n_total)) if 0 < self.step < 1 else int(self.step)
        support = np.ones(n_total, dtype=bool)
        ranking = np.ones(n_total, dtype=int)
        removed = []           # each round's removed features, weakest first
        kept = np.arange(n_total)  # the survivors, by weakest weight first
        gram = X @ X.T

        while support.sum() > self.n_features:
            features = np.flatnonzero(support)
            svm = SVC(kernel='precomputed', C=self.C).fit(gram, y)
            # The SVM's weights are X^T alpha, alpha the signed dual coefficients.
            alpha = np.zeros(n_samples)
            alpha[svm.support_] = svm.dual_coef_[0]
            order = np.argsort((alpha @ X)[features] ** 2)
            cut = min(step, len(features) - self.n_features)
            dropped, kept = features[order[:cut]], features[order[cut:]]
            removed.append(dropped)
            support[dropped] = False
            ranking[~support] += 1
            # Update the Gram matrix from whichever side is smaller.
            if len(dropped) < len(kept):
                gram -= X[:, dropped] @ X[:, dropped].T
            else:
                gram = X[:, support] @ X[:, support].T

        elimination = np.concatenate([*removed, kept])
        self.feature_importances_ = np.empty(n_total)
        self.feature_importances_[elimination] = np.arange(n_total)
        self.ranking_ = ranking
        self.support_ = support
        return self
