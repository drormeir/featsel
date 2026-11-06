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
"""

import numpy as np
from scipy.special import digamma
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state


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


# Features handled per batch: the batch's sorted copies and neighbor
# distances are a few times n_samples * _BATCH float64 values.
_BATCH = 1024


def mutual_info_classif(X, y, n_neighbors=3, random_state=None):
    """
    Mutual information of each feature with a discrete target, in nats.

    Equal to sklearn.feature_selection.mutual_info_classif(X, y,
    n_neighbors=n_neighbors, random_state=random_state) with continuous
    features.
    """
    X = scale(np.array(X, dtype=np.float64), with_mean=False, copy=False)
    rng = check_random_state(random_state)
    X += 1e-10 * np.maximum(1, np.mean(np.abs(X), axis=0)) * rng.standard_normal(X.shape)

    _, codes, counts = np.unique(y, return_inverse=True, return_counts=True)
    n_class = counts[codes]
    used = n_class > 1                       # scikit-learn ignores single-member classes
    k_all = np.minimum(n_neighbors, n_class - 1)
    constant = (digamma(used.sum()) + np.mean(digamma(k_all[used]))
                - np.mean(digamma(n_class[used])))

    mi = np.empty(X.shape[1])
    for g0 in range(0, X.shape[1], _BATCH):
        block = X[used, g0:g0 + _BATCH]
        radius = np.empty_like(block)
        for label in np.flatnonzero(counts > 1):
            members = codes[used] == label
            radius[members] = _kth_neighbor_distance(block[members], k_all[used][members][0])
        radius = np.nextafter(radius, 0)
        m = _counts_within(block, radius)
        # Each feature's mean over a contiguous row, summed pairwise as
        # NumPy sums the one-dimensional array scikit-learn averages.
        mean_m = np.ascontiguousarray(digamma(m).T).mean(axis=1)
        mi[g0:g0 + _BATCH] = np.maximum(0, constant - mean_m)
    return mi


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
