"""
Higher Criticism thresholding: keep the features whose F-test p-values fall
below the threshold that Higher Criticism chooses.
"""

import numpy as np
from sklearn.feature_selection import f_classif

from ..base import FeatureSelector
from .log_pvalue import log_f_sf


class HigherCriticismSelector(FeatureSelector):
    """
    Select features by Higher Criticism thresholding (Donoho and Jin).

    Computes one p-value per feature with the ANOVA F-test, sorts them,
    p(1) <= ... <= p(N), and scores each rank i by how far p(i) falls below
    its expected value under the null:

        '2004': HC(i) = sqrt(N) (i/N - p(i)) / sqrt(p(i) (1 - p(i)))
                Donoho and Jin (2004), Annals of Statistics, section 1.2.
        '2008': HC(i) = sqrt(N) (mu(i) - p(i)) / sqrt(mu(i) (1 - mu(i))),
                mu(i) = i / (N + 1)
                The HC thresholding of Donoho and Jin (2008), with each
                ordered p-value centred and scaled by its exact null mean,
                i/(N+1), the mean of Beta(i, N-i+1), in place of the
                approximation i/N of their eq. 1.1.

    The maximum is taken over 1 <= i <= alpha0 N. Every feature with a p-value
    at or below p(i) at the maximum is kept, so the selector picks its own
    count.

    p-values are handled as logs (see log_pvalue.py): on strong data many are
    below 1e-308, where a float64 p-value is 0 and the 2004 form would divide
    by zero.

    Best for: many weak signals among many null features (rare/weak setting)
    Pros: fast, no count to choose, adapts the count to the data
    Cons: same blind spots as the F-test; the 2004 form can stop at a few
    features when the strongest p-values are extremely small

    Parameters
    ----------
    variant : {'2008', '2004'}, default='2008'
        Which HC objective to maximize.
    alpha0 : float, default=0.1
        Fraction of the smallest p-values the maximum searches, as in both
        papers.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    scores_ : np.ndarray
        F statistic for each feature.
    log_pvalues_ : np.ndarray
        Natural log of each feature's p-value. A constant feature gets 0.
    feature_importances_ : np.ndarray
        Alias for scores_.

    Examples
    --------
    >>> from featsel.selectors import HigherCriticismSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_features=50, n_informative=5,
    ...                            n_redundant=0, shuffle=False, random_state=0)
    >>> # Features 0-4 are informative.
    >>> HigherCriticismSelector().fit(X, y).get_support(indices=True).tolist()
    [0, 1, 2, 4]
    """

    def __init__(self, variant='2008', alpha0=0.1, check_input=True):
        self.variant = variant
        self.alpha0 = alpha0
        self.check_input = check_input

    def fit(self, X, y):
        """
        Keep every feature at or below the Higher Criticism threshold.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : HigherCriticismSelector
            Fitted selector.
        """
        if self.variant not in ('2004', '2008'):
            raise ValueError(f"variant must be '2004' or '2008', got '{self.variant}'")
        X, y = self._validate(X, y)

        n_classes = len(np.unique(y))
        self.scores_, _ = f_classif(X, y)
        self.feature_importances_ = self.scores_
        log_p = log_f_sf(self.scores_, n_classes - 1, len(y) - n_classes)
        self.log_pvalues_ = np.where(np.isnan(log_p), 0.0, log_p)

        n = self.n_features_in_
        order = np.argsort(self.log_pvalues_, kind='stable')
        i = np.arange(1, max(1, int(self.alpha0 * n)) + 1)
        log_p_sorted = self.log_pvalues_[order[:len(i)]]
        p_sorted = np.exp(log_p_sorted)

        if self.variant == '2004':
            center = i / n
            log_var = log_p_sorted + np.log1p(-p_sorted)
        else:
            center = i / (n + 1)
            log_var = np.log(center) + np.log1p(-center)

        # Compared as logs, since the 2004 objective overflows when p(i) is tiny.
        # A rank with p(i) at or above the center scores <= 0 and can win only
        # when no rank scores above 0.
        above = center > p_sorted
        if above.any():
            with np.errstate(divide='ignore'):
                log_hc = np.log(center - p_sorted) - log_var / 2
            best = np.argmax(np.where(above, log_hc, -np.inf))
        else:
            best = np.argmax((center - p_sorted) / np.exp(log_var / 2))

        self.support_ = self.log_pvalues_ <= log_p_sorted[best]
        return self
