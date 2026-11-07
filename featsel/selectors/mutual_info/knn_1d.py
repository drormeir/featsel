"""
scikit-learn's mutual_info_classif, computed by sorting instead of KD-trees.

For a continuous feature and a discrete target, scikit-learn estimates the
mutual information as Ross (2014) does. Each sample's radius is its distance
to its k-th nearest neighbor of the same class, just below. m is the number
of samples of any class within that radius, the sample itself included.
Then

    MI = max(0, digamma(n) + mean(digamma(k)) - mean(digamma(n_class)) - mean(digamma(m)))

over the samples whose class has more than one member.

A feature is one-dimensional, so a KD-tree is not needed. In a class's
sorted values a sample's k nearest neighbors lie among the k on each side,
and the samples within a radius are one contiguous run of the feature's
sorted values. Distances are the same float subtractions the trees make,
and the means run over the samples in the same order, so the result is
scikit-learn's exactly.

The preparation is scikit-learn's too: each feature is divided by its
standard deviation, then noise of 1e-10 times max(1, mean |x|) is added,
drawn from random_state as one (n_samples, n_features) matrix.

On a GPU all of it runs in PyTorch, each sum in NumPy's order, except the
noise: it must come from NumPy's generator to be scikit-learn's.
"""

import numpy as np
from scipy.special import digamma
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state

from .. import gpu


def matches_sklearn(y, n_neighbors):
    """
    True when this module gives scikit-learn's result exactly.

    For a class where k >= its size // 2, scikit-learn finds neighbors by
    brute force, with Euclidean distances from a dot-product formula that
    rounds differently from a subtraction. With k = 3 that is a class of 2
    to 7 members. Single-member classes are ignored by both.
    """
    counts = np.unique(y, return_counts=True)[1]
    counts = counts[counts > 1]
    return bool(np.all(np.minimum(n_neighbors, counts - 1) < counts // 2))


# Features handled per batch on the CPU: the batch's sorted copies and
# neighbor distances are a few times n_samples * _BATCH float64 values.
_BATCH = 1024

# Peak bytes per value of a batch on a GPU: the batch, its noise, its sorted
# copy and sort order, the radii, the shifted differences and the search ends.
_GPU_BYTES_PER_VALUE = 128


def mutual_info_classif(X, y, n_neighbors=3, random_state=None, device='cpu'):
    """
    Mutual information of each feature with a discrete target, in nats.

    Equal to sklearn.feature_selection.mutual_info_classif(X, y,
    n_neighbors=n_neighbors, random_state=random_state) with continuous
    features.

    device is where it runs: 'cpu' in NumPy, or a PyTorch device such as
    'cuda', in float64, batched over features. On a device X stays there:
    the scaling, the neighbor searches, digamma and the means all run there,
    each sum in NumPy's order, so the result is the same, bit for bit. Only
    the noise is drawn on the CPU, from scikit-learn's random stream, and
    sent over. X is a NumPy array, an array-like, or (on a device) a tensor.
    """
    rng = check_random_state(random_state)
    _, codes, counts = np.unique(y, return_inverse=True, return_counts=True)
    n_class = counts[codes]
    used = n_class > 1                       # scikit-learn ignores single-member classes
    k_all = np.minimum(n_neighbors, n_class - 1)
    constant = (digamma(used.sum()) + np.mean(digamma(k_all[used]))
                - np.mean(digamma(n_class[used])))
    # Each class's members among the used samples, and its k.
    groups = [(codes[used] == label, min(n_neighbors, counts[label] - 1))
              for label in np.flatnonzero(counts > 1)]
    # digamma of every count a sample can have, 1 to all used samples.
    table = digamma(np.arange(1, used.sum() + 1, dtype=np.float64))

    if device == 'cpu':
        X = scale(np.array(X, dtype=np.float64), with_mean=False, copy=False)
        X += 1e-10 * np.maximum(1, np.mean(np.abs(X), axis=0)) * rng.standard_normal(X.shape)
        mean_m = _mean_digamma_numpy(X, used, groups, table)
    else:
        mean_m = _mean_digamma_torch(X, rng.standard_normal(X.shape), used, groups, table,
                                     device)
    return np.maximum(0, constant - mean_m)


def _mean_digamma_numpy(X, used, groups, table):
    """Each feature's mean of digamma(m) over the used samples, in NumPy."""
    mean_m = np.empty(X.shape[1])
    for g0 in range(0, X.shape[1], _BATCH):
        block = X[used, g0:g0 + _BATCH]
        radius = np.empty_like(block)
        for members, k in groups:
            radius[members] = _kth_neighbor_distance(block[members], k)
        m = _counts_within(block, np.nextafter(radius, 0))
        # Each feature's mean over a contiguous row, summed pairwise as
        # NumPy sums the one-dimensional array scikit-learn averages.
        mean_m[g0:g0 + _BATCH] = table[np.ascontiguousarray(m.T) - 1].mean(axis=1)
    return mean_m


def _mean_digamma_torch(X, noise, used, groups, table, device):
    """
    Each feature's mean of digamma(m) over the used samples, on a PyTorch device.

    X is scaled and given its noise there, batch by batch, as scikit-learn
    does it to a float64 copy of X.
    """
    torch = gpu.import_torch(device)
    if not isinstance(X, torch.Tensor):
        X = np.asarray(X)
    strides = X.stride() if isinstance(X, torch.Tensor) else X.strides
    # NumPy sums down each feature of the float64 copy pairwise when the copy
    # holds a feature's values contiguous (one feature, or column-major X),
    # else one sample after another.
    feature_sum = (_pairwise_sum if X.shape[1] == 1 or strides[0] < strides[1]
                   else _sequential_sum)
    used_t = None if used.all() else torch.as_tensor(np.flatnonzero(used), device=device)
    members_t = [(torch.as_tensor(np.flatnonzero(members), device=device), k)
                 for members, k in groups]
    table_t = torch.as_tensor(table, device=device)
    zero = torch.zeros((), dtype=torch.float64, device=device)
    mean_m = np.empty(X.shape[1])
    g0 = 0
    for block in gpu.float64_columns(torch, X, device, _GPU_BYTES_PER_VALUE):
        g1 = g0 + block.shape[1]
        # Features as rows, so each feature's samples are contiguous.
        rows = _scale_and_noise(torch, block.T.contiguous(),
                                torch.as_tensor(noise[:, g0:g1], device=device).T, feature_sum)
        del block
        if used_t is not None:
            rows = rows[:, used_t]
        radius = torch.empty_like(rows)
        for members, k in members_t:
            radius[:, members] = _kth_neighbor_distance_torch(torch, rows[:, members], k)
        m = _counts_within_torch(torch, rows, torch.nextafter(radius, zero))
        del rows, radius
        mean_m[g0:g1] = (_pairwise_sum(table_t[m - 1]) / _exact(torch, m.shape[1], m)).cpu().numpy()
        g0 = g1
    return mean_m


def _scale_and_noise(torch, rows, noise, feature_sum):
    """
    scikit-learn's preparation of each row (a feature), as it runs in NumPy.

    scale(with_mean=False) divides by np.nanstd, a standard deviation of
    near zero counting as 1; then noise times 1e-10 * max(1, mean |x|) is
    added. Every sum over samples is feature_sum, NumPy's order.
    """
    n = _exact(torch, rows.shape[1], rows)
    centered = rows - (feature_sum(rows) / n)[:, None]
    std = torch.sqrt(feature_sum(centered * centered) / n)
    del centered
    std[std < 10 * np.finfo(np.float64).eps] = 1.0
    rows = rows / std[:, None]
    rows += (1e-10 * torch.clamp_min(feature_sum(rows.abs()) / n, 1))[:, None] * noise
    return rows


def _exact(torch, number, like):
    """
    number as a float64 tensor on like's device, to divide by.

    PyTorch on a GPU divides by a Python number as a product with its
    reciprocal, which rounds differently from NumPy's division. A tensor
    divisor gives a true division.
    """
    return torch.tensor(float(number), dtype=torch.float64, device=like.device)


def _sequential_sum(rows):
    """Each row's sum, one value after another, as NumPy sums down a C-order column."""
    total = rows[:, 0].clone()
    for i in range(1, rows.shape[1]):
        total += rows[:, i]
    return total


def _pairwise_sum(rows, lo=0, n=None):
    """
    Each row's sum over columns lo to lo + n, in NumPy's pairwise order.

    NumPy's pairwise_sum: fewer than 8 values one after another; up to 128 in
    8 interleaved partial sums, combined as a tree, then the remainder; more
    split in two at a multiple of 8.
    """
    if n is None:
        n = rows.shape[1]
    if n < 8:
        total = rows.new_zeros(len(rows))
        for i in range(lo, lo + n):
            total += rows[:, i]
        return total
    if n <= 128:
        r = rows[:, lo:lo + 8].clone()
        i = 8
        while i < n - n % 8:
            r += rows[:, lo + i:lo + i + 8]
            i += 8
        total = ((r[:, 0] + r[:, 1]) + (r[:, 2] + r[:, 3])) + ((r[:, 4] + r[:, 5]) + (r[:, 6] + r[:, 7]))
        for j in range(i, n):
            total += rows[:, lo + j]
        return total
    half = n // 2
    half -= half % 8
    return _pairwise_sum(rows, lo, half) + _pairwise_sum(rows, lo + half, n - half)


def _kth_neighbor_distance(values, k):
    """
    Each sample's distance to its k-th nearest other sample, per column.

    In sorted order the k nearest are among the k on each side. With L_a
    the distance to the a-th on the left and R_b to the b-th on the right,
    both increasing, the k-th smallest of all is min over a of
    max(L_a, R_(k - a)), with L_0 = R_0 = 0 and a missing neighbor at
    infinity.
    """
    n = len(values)
    order = np.argsort(values, axis=0, kind='stable')
    s = np.take_along_axis(values, order, axis=0)
    padded = np.full((n + 2 * k, s.shape[1]), np.inf)
    padded[k:k + n] = s
    left = [np.zeros_like(s)] + [s - padded[k - a:k - a + n] for a in range(1, k + 1)]
    right = [np.zeros_like(s)] + [padded[k + b:k + b + n] - s for b in range(1, k + 1)]
    # A missing left neighbour gives s - inf = -inf; it must count as infinitely far.
    left = [np.where(np.isneginf(d), np.inf, d) for d in left]
    kth = np.minimum.reduce([np.maximum(left[a], right[k - a]) for a in range(k + 1)])
    out = np.empty_like(kth)
    np.put_along_axis(out, order, kth, axis=0)
    return out


def _counts_within(values, radius):
    """
    Per column, how many values lie within each sample's radius of it,
    |values[j] - values[i]| <= radius[i], the sample itself included.

    The run of such values is contiguous in sorted order. A binary search
    for values[i] -/+ radius[i] finds its ends to within rounding; each end
    is then moved until the exact test above holds, as the KD-tree tests it.
    """
    s = np.sort(values, axis=0)
    lo = np.empty(values.shape, dtype=np.int64)
    hi = np.empty(values.shape, dtype=np.int64)
    for j in range(values.shape[1]):   # NumPy's searchsorted takes one sorted array
        lo[:, j] = np.searchsorted(s[:, j], values[:, j] - radius[:, j], side='left')
        hi[:, j] = np.searchsorted(s[:, j], values[:, j] + radius[:, j], side='right')

    def inside(index):
        """Whether sorted position index (clipped) is within each sample's radius."""
        at = np.take_along_axis(s, np.clip(index, 0, len(s) - 1), axis=0)
        return np.abs(at - values) <= radius

    n = len(s)
    for _ in range(n):   # rounding moves an end by a position or two at most
        step_lo = np.where((lo > 0) & inside(lo - 1), -1,
                           np.where((lo < n) & ~inside(lo), 1, 0))
        step_hi = np.where((hi < n) & inside(hi), 1,
                           np.where((hi > 0) & ~inside(hi - 1), -1, 0))
        if not (step_lo.any() or step_hi.any()):
            break
        lo += step_lo
        hi += step_hi
    return hi - lo



def _kth_neighbor_distance_torch(torch, values, k):
    """
    _kth_neighbor_distance on a PyTorch device, with features as rows.

    The same subtractions, minima and maxima. With L_0 = R_0 = 0 the terms
    a = 0 and a = k are R_k and L_k. A missing left neighbor is padded with
    -inf, so its distance is inf at once.
    """
    n = values.shape[1]
    s, order = torch.sort(values, dim=1, stable=True)
    pad = s.new_full((len(s), k), torch.inf)
    padded = torch.cat([-pad, s, pad], dim=1)

    def left(a):
        return s - padded[:, k - a:k - a + n]

    def right(b):
        return padded[:, k + b:k + b + n] - s

    kth = torch.minimum(left(k), right(k))
    for a in range(1, k):
        kth = torch.minimum(kth, torch.maximum(left(a), right(k - a)))
    return torch.empty_like(kth).scatter_(1, order, kth)


def _counts_within_torch(torch, values, radius):
    """_counts_within on a PyTorch device, with features as rows."""
    s = torch.sort(values, dim=1).values
    lo = torch.searchsorted(s, values - radius, side='left')
    hi = torch.searchsorted(s, values + radius, side='right')

    def inside(index):
        """Whether sorted position index (clipped) is within each sample's radius."""
        at = s.gather(1, index.clamp(0, n - 1))
        return (at - values).abs() <= radius

    n = s.shape[1]
    for _ in range(n):   # rounding moves an end by a position or two at most
        step_lo = torch.where((lo > 0) & inside(lo - 1), -1,
                              torch.where((lo < n) & ~inside(lo), 1, 0))
        step_hi = torch.where((hi < n) & inside(hi), 1,
                              torch.where((hi > 0) & ~inside(hi - 1), -1, 0))
        if not (step_lo.any() | step_hi.any()):
            break
        lo += step_lo
        hi += step_hi
    return hi - lo
