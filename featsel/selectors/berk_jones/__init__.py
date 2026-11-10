"""
Berk-Jones thresholding: keep the features whose Mann-Whitney U p-values
fall below the threshold where the exact Berk-Jones score is smallest.
"""

import numpy as np
from scipy.special import betainc, betaln

from ..base import FeatureSelector
from ..log_pvalue import log_mannwhitneyu

# Below this, betainc's result loses precision on its way to underflow, so
# the series takes over.
_SERIES_BELOW = np.log(1e-300)


def log_beta_cdf(log_x, a, b):
    """
    log I_x(a, b), the log of the Beta(a, b) CDF at x = exp(log_x).

    Finite even when the CDF is far below the smallest float64, as it is for
    the strongest p-values. Where SciPy's betainc is at least 1e-300 it is
    used as is. Below that, the series of DLMF 8.17.8:

        I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * F(a + b, 1; a + 1; x)

    in log form. Its terms shrink by x (a + b + n) / (a + 1 + n), so it
    converges fast where the CDF is this small, far below the mean a / (a + b).
    """
    log_x, a, b = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (log_x, a, b)))
    x = np.exp(log_x)
    with np.errstate(divide='ignore'):
        result = np.array(np.log(betainc(a, b, x)))
    series = result < _SERIES_BELOW
    if series.any():
        xs, as_, bs = x[series], a[series], b[series]
        term, total = np.ones_like(xs), np.ones_like(xs)
        n = 0
        while (term > 1e-17 * total).any():
            if n == 100_000:
                raise RuntimeError('the incomplete beta series did not converge')
            term = term * xs * (as_ + bs + n) / (as_ + 1 + n)
            total += term
            n += 1
        result[series] = (as_ * log_x[series] + bs * np.log1p(-xs) - np.log(as_)
                          - betaln(as_, bs) + np.log(total))
    return result


class BerkJonesSelector(FeatureSelector):
    """
    Select features by exact Berk-Jones thresholding.

    Computes one p-value per feature with the two-sided Mann-Whitney U test
    and sorts them, p(1) <= ... <= p(N). Under the null, p(i) follows
    Beta(i, N - i + 1). Each rank i is scored by that distribution's CDF at
    the observed p-value:

        BJ(i) = P(Beta(i, N - i + 1) <= p(i))

    the chance, under the null, of an i-th smallest p-value this small. This
    is the one-sided exact Berk-Jones statistic M_n^+ of Moscovich, Nadler and
    Spiegelman (2016), after Berk and Jones (1979). The minimum is taken over
    1 <= i <= alpha0 N, as for HigherCriticismSelector, and every feature
    with a p-value at or below p(i) at the minimum is kept. So the selector
    picks its own count.

    Higher Criticism standardizes p(i) by a mean and a standard deviation.
    Berk-Jones uses p(i)'s whole null distribution instead.

    p-values and the CDF are handled as logs (see selectors/log_pvalue/):
    on strong data many p-values are below 1e-308, where a float64 p-value is
    0, and the CDF is smaller still.

    Best for: many weak signals among many null features (rare/weak setting)
    Pros: no count to choose; exact in the tail, where HC approximates
    Cons: one feature at a time, so correlated features all pass together

    Parameters
    ----------
    alpha0 : float, default=0.1
        Fraction of the smallest p-values the minimum searches, as for
        HigherCriticismSelector.
    device : str, default='auto'
        Where the U test ranks: 'auto' uses a CUDA GPU when one can be used
        (the `gpu` extra), else NumPy; 'cpu' always NumPy; 'cuda' always the
        GPU, waiting while another fit holds it and raising if it cannot run.
        The selection is identical on every device.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    device_ : str
        The device the U test ran on: 'cpu' or 'cuda'.
    auc_ : np.ndarray
        AUC of each feature for the second class in sorted order.
    log_pvalues_ : np.ndarray
        Natural log of each feature's p-value. A constant feature gets 0.

    Examples
    --------
    >>> from featsel.selectors import BerkJonesSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_features=50, n_informative=5,
    ...                            n_redundant=0, shuffle=False, random_state=0)
    >>> # Features 0-4 are informative. Feature 1's p-value, 2.5e-20, is so far
    >>> # below the rest that rank 1 scores lowest, so it is kept alone.
    >>> BerkJonesSelector().fit(X, y).get_support(indices=True).tolist()
    [1]
    """

    def __init__(self, alpha0=0.1, device='auto', check_input=True):
        self.alpha0 = alpha0
        self.device = device
        self.check_input = check_input

    def _fit(self, X, y):
        """
        Keep every feature at or below the Berk-Jones threshold.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Class labels, exactly two classes.

        Returns
        -------
        self : BerkJonesSelector
            Fitted selector.
        """
        X, y = self._validate(X, y, keep_gpu_tensor=True, check_input=self.check_input)

        self.log_pvalues_, self.auc_, self.device_ = log_mannwhitneyu(
            X, y, self.device, return_device=True)

        n = self.n_features_in_
        order = np.argsort(self.log_pvalues_, kind='stable')
        i = np.arange(1, max(1, int(self.alpha0 * n)) + 1)
        log_p_sorted = self.log_pvalues_[order[:len(i)]]

        best = np.argmin(log_beta_cdf(log_p_sorted, i, n - i + 1))
        self.support_ = self.log_pvalues_ <= log_p_sorted[best]
        return self
