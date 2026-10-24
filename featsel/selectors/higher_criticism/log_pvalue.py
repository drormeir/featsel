"""
Log p-values of the F-test that stay finite when the p-value is below 1e-308.

scipy's f.sf returns 0 for such p-values, and f.logsf returns -inf, because it
takes the log after computing p. Here the log is taken before anything is
multiplied together, so no intermediate value underflows.
"""

import numpy as np
from scipy.special import betaln

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
