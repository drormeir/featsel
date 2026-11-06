"""
Log p-values that stay finite when the p-value is below 1e-308.

A float64 p-value below 1e-308 is stored as 0. Here the log is taken before
anything is multiplied together, so no intermediate value underflows.

- log_mannwhitneyu: the Mann-Whitney U test, per feature. Used by
  MannWhitneySelector and HigherCriticismSelector. Its ranking runs in
  PyTorch on a GPU when one can be used, else in NumPy, with identical results.
- log_f_sf: the F-test, kept to compare its p-values with the U test's.
  scipy's f.sf returns 0 below 1e-308, and f.logsf returns -inf, because it
  takes the log after computing p.
"""

import sys

import numpy as np
from scipy.special import betaln
from scipy.stats import norm

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


def log_mannwhitneyu(X, y, device='auto'):
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
    X : np.ndarray or torch.Tensor of shape (n_samples, n_features)
        A tensor on a GPU is ranked there, whatever `device` says, without
        a copy to the CPU. A tensor on the CPU is treated as a NumPy array.
    y : np.ndarray or torch.Tensor of shape (n_samples,)
        Exactly two classes. The second in sorted order is the one whose AUC
        is reported.
    device : str, default='auto'
        Where the ranking runs. 'auto' uses a CUDA GPU through PyTorch when
        one can be used, else NumPy. 'cpu' always uses NumPy. Any other
        PyTorch device, such as 'cuda', always uses PyTorch on it and raises
        if it cannot. Outputs are NumPy arrays on every device, and the
        results are identical.

    Returns
    -------
    log_p : np.ndarray of shape (n_features,)
        log p per feature; 0 (p = 1) for a feature with a single value.
    auc : np.ndarray of shape (n_features,)
    """
    torch = sys.modules.get('torch')   # a tensor exists only once PyTorch is imported
    if torch is not None and isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if torch is not None and isinstance(X, torch.Tensor):
        if X.device.type == 'cpu':
            X = X.numpy()
        else:
            device = str(X.device)

    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError(f"The Mann-Whitney U test needs two classes, got {len(labels)}")
    second = y == labels[1]
    n2 = second.sum()
    n1 = len(y) - n2
    n = n1 + n2

    if device == 'auto':
        on_gpu = _u_and_ties_on_gpu_if_possible(X, second)
        u, ties = on_gpu if on_gpu is not None else _u_and_ties(X, second)
    elif device == 'cpu':
        u, ties = _u_and_ties(X, second)
    else:
        u, ties = _u_and_ties_torch(X, second, device)

    # The rest is per feature and cheap. It stays in NumPy for every device,
    # so the results do not depend on the device.
    sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))))
    with np.errstate(divide='ignore', invalid='ignore'):  # a constant feature has sigma 0
        z = (np.abs(u - n1 * n2 / 2) - 0.5) / sigma
    log_p = np.minimum(np.log(2) + norm.logsf(z), 0.0)
    return log_p, u / (n1 * n2)


def _u_and_ties(X, second):
    """
    U of the second class, and the tie term sum(t^3 - t), per feature, in NumPy.

    One sort per feature gives both the average ranks and the tie groups.
    """
    n = len(second)
    n2 = second.sum()
    # Each feature is a row of Xt, so the sort runs over contiguous memory.
    Xt = np.ascontiguousarray(X.T)
    order = np.argsort(Xt, axis=1)
    values = np.take_along_axis(Xt, order, axis=1)
    del Xt

    # In sorted order, a tie group runs from position `start` to `end`.
    first = np.empty(values.shape, dtype=bool)
    first[:, 0] = True
    np.not_equal(values[:, 1:], values[:, :-1], out=first[:, 1:])
    del values
    last = np.empty_like(first)
    last[:, -1] = True
    last[:, :-1] = first[:, 1:]
    position = np.arange(n, dtype=np.int32)
    start = np.maximum.accumulate(np.where(first, position, 0), axis=1)
    end = np.minimum.accumulate(np.where(last, position, n)[:, ::-1], axis=1)[:, ::-1]
    del first, last

    # Average rank, 1-based: (start + end) / 2 + 1. Summed over the second
    # class, in float64 so the sum is exact.
    rank_sum = np.where(second[order], start + end, 0).sum(axis=1, dtype=np.float64) / 2 + n2
    u = rank_sum - n2 * (n2 + 1) / 2

    # A group of t tied values adds t^3 - t; each of its t members adds t^2 - 1.
    t = end - start + 1
    ties = (np.square(t, dtype=np.float64) - 1).sum(axis=1)
    return u, ties


# Features ranked per batch on a PyTorch device. At 2,678 samples a batch
# peaks near 420 MB.
_TORCH_BATCH = 4096


def _u_and_ties_on_gpu_if_possible(X, second):
    """
    U and the tie term on a CUDA GPU, or None when no GPU can be used.

    None when PyTorch is missing or cannot load (under a tight address-space
    cap, such as the runner's per-worker memory cap, it fails to map its
    libraries), when no GPU is present, or when the GPU runs out of memory
    because other processes share it.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return _u_and_ties_torch(X, second, 'cuda')
    except RuntimeError as error:
        # torch.OutOfMemoryError, or a raw CUDA out-of-memory error
        # (torch.AcceleratorError in recent PyTorch); both say this.
        if 'out of memory' not in str(error):
            raise
    torch.cuda.empty_cache()
    return None


def _u_and_ties_torch(X, second, device):
    """
    The same as _u_and_ties, in PyTorch on `device`, _TORCH_BATCH features at a time.

    U and the tie term are sums of whole and half numbers, exact in float64
    on any device, so they equal the NumPy result exactly. Afterwards the
    GPU memory PyTorch cached is released, so an idle worker holds little.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError(f"device={device!r} could not import PyTorch. "
                          "If it is missing: pip install -e '.[gpu]'") from error

    second_t = torch.as_tensor(second, device=device)
    position = torch.arange(len(second), dtype=torch.int32, device=device)
    u = np.empty(X.shape[1])
    ties = np.empty(X.shape[1])
    for g0 in range(0, X.shape[1], _TORCH_BATCH):
        g1 = g0 + _TORCH_BATCH
        block = X[:, g0:g1].T
        if isinstance(block, np.ndarray):
            block = np.ascontiguousarray(block)
        # Passed straight in, so the batch can free it once sorted.
        u[g0:g1], ties[g0:g1] = _u_and_ties_batch(
            torch, torch.as_tensor(block, device=device).contiguous(), second_t, position)
    del second_t, position
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return u, ties


def _u_and_ties_batch(torch, block, second, position):
    """U and the tie term of one batch, features as rows, as NumPy arrays."""
    n = block.shape[1]
    n2 = int(second.sum())
    values, order = torch.sort(block, dim=1)
    del block

    first = torch.empty(values.shape, dtype=torch.bool, device=values.device)
    first[:, 0] = True
    torch.ne(values[:, 1:], values[:, :-1], out=first[:, 1:])
    del values
    last = torch.empty_like(first)
    last[:, -1] = True
    last[:, :-1] = first[:, 1:]
    start = torch.cummax(torch.where(first, position, 0), dim=1).values
    end = torch.cummin(torch.where(last, position, n).flip(1), dim=1).values.flip(1)
    del first, last

    rank_sum = torch.where(second[order], start + end, 0).sum(1, dtype=torch.float64)
    u = rank_sum / 2 + n2 - n2 * (n2 + 1) / 2
    t = (end - start + 1).to(torch.float64)
    return u.cpu().numpy(), (t * t - 1).sum(1).cpu().numpy()
