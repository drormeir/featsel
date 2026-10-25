"""
Log p-values that stay finite when the p-value is below 1e-308.

A float64 p-value below 1e-308 is stored as 0. Here the log is taken before
anything is multiplied together, so no intermediate value underflows.

- log_mannwhitneyu: the Mann-Whitney U test, per feature. Used by
  HigherCriticismSelector.
- log_f_sf: the F-test, kept to compare its p-values with the U test's.
  scipy's f.sf returns 0 below 1e-308, and f.logsf returns -inf, because it
  takes the log after computing p.
"""

import numpy as np
from scipy.special import betaln
from scipy.stats import norm, rankdata

# Continued-fraction settings, as in Numerical Recipes' betacf.
_MAX_TERMS = 1000
_EPS = 1e-15
_TINY = 1e-300


def _beta_continued_fraction(a, b, x):
    """
    The continued fraction of the incomplete beta function, by Lentz's method.

    Converges quickly where x < (a + 1) / (a + b + 2). Its value is of order
    one, so it cannot underflow. Each element stops once it has converged:
    further terms only add rounding noise.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = 1.0 / np.where(np.abs(d) < _TINY, _TINY, d)
    h = d.copy()
    done = np.zeros(x.shape, dtype=bool)

    for m in range(1, _MAX_TERMS + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / np.where(np.abs(d) < _TINY, _TINY, d)
            c = 1.0 + aa / c
            c = np.where(np.abs(c) < _TINY, _TINY, c)
            step = d * c
            h = np.where(done, h, h * step)
        done |= np.abs(step - 1.0) < _EPS
        if done.all():
            return h

    raise RuntimeError(f"Incomplete beta continued fraction did not converge in "
                       f"{_MAX_TERMS} terms")


def _log_incomplete_beta(a, b, log_x, log_1mx):
    """
    log I_x(a, b), the regularized incomplete beta, for x < (a + 1) / (a + b + 2).

    I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * continued fraction.
    """
    x = np.exp(log_x)
    return (a * log_x + b * log_1mx - betaln(a, b) - np.log(a)
            + np.log(_beta_continued_fraction(a, b, x)))


def log_f_sf(f, dfn, dfd):
    """
    Natural log of the F-test p-value, P(F(dfn, dfd) > f), for each f.

    The p-value is the regularized incomplete beta I_x(dfd/2, dfn/2) at
    x = dfd / (dfd + dfn f). Small p-values are computed from that series in
    log form; large ones as log(1 - I_{1-x}(dfn/2, dfd/2)).

    With two classes, dfn = 1 and the result is the exact log p-value of the
    two-sided pooled t-test, since F = t^2.

    Parameters
    ----------
    f : array-like
        F statistics, as from sklearn's f_classif. NaN stays NaN.
    dfn : float
        Numerator degrees of freedom: number of classes - 1.
    dfd : float
        Denominator degrees of freedom: number of samples - number of classes.

    Returns
    -------
    np.ndarray
        log p for each f: 0 at f = 0, -inf at f = inf.
    """
    f = np.asarray(f, dtype=float)
    a, b = dfd / 2.0, dfn / 2.0
    ratio = dfn * f / dfd
    log_x = -np.log1p(ratio)                  # log(dfd / (dfd + dfn f))
    with np.errstate(divide='ignore', invalid='ignore'):  # f = 0 or inf, set below
        log_1mx = np.log(ratio) + log_x       # log(dfn f / (dfd + dfn f))

    out = np.full(f.shape, np.nan)
    out[f == 0] = 0.0
    out[f == np.inf] = -np.inf

    inner = np.isfinite(f) & (f > 0)
    tail = inner & (np.exp(log_x) < (a + 1.0) / (a + b + 2.0))
    head = inner & ~tail

    out[tail] = _log_incomplete_beta(a, b, log_x[tail], log_1mx[tail])
    out[head] = np.log1p(-np.exp(_log_incomplete_beta(b, a, log_1mx[head], log_x[head])))
    return out


def log_mannwhitneyu(X, y):
    """
    Natural log of the two-sided Mann-Whitney U p-value of each feature, and its AUC.

    The test uses only the order of each feature's values, so any monotone
    transform of a feature leaves it unchanged. U counts the pairs, one sample
    from each class, in which the second class's sample has the higher value;
    ties count one half. AUC = U / (n1 n2).

    The p-value is the normal approximation with tie correction and
    continuity correction, as scipy.stats.mannwhitneyu(method='asymptotic')
    computes it. Its log comes from norm.logsf, which stays finite far below
    1e-308. Tied values get their average rank.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
        Exactly two classes. The second in sorted order is the one whose AUC
        is reported.

    Returns
    -------
    log_p : np.ndarray of shape (n_features,)
        log p per feature; 0 (p = 1) for a feature with a single value.
    auc : np.ndarray of shape (n_features,)
    """
    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError(f"The Mann-Whitney U test needs two classes, got {len(labels)}")
    second = y == labels[1]
    n2 = second.sum()
    n1 = len(y) - n2
    n = n1 + n2

    low = rankdata(X, method='min', axis=0)
    high = rankdata(X, method='max', axis=0)
    u = ((low + high)[second] / 2).sum(axis=0) - n2 * (n2 + 1) / 2

    # A group of t tied values adds t^3 - t; each of its t members adds t^2 - 1.
    ties = ((high - low + 1) ** 2 - 1).sum(axis=0)
    sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))))

    with np.errstate(divide='ignore', invalid='ignore'):  # a constant feature has sigma 0
        z = (np.abs(u - n1 * n2 / 2) - 0.5) / sigma
    log_p = np.minimum(np.log(2) + norm.logsf(z), 0.0)
    return log_p, u / (n1 * n2)
