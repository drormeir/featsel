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

# Peak bytes per value of a batch on a GPU: the batch, its sorted copy and
# sort order, the radii, the shifted differences and the search ends.
_GPU_BYTES_PER_VALUE = 128


def mutual_info_classif(X, y, n_neighbors=3, random_state=None, device='cpu'):
    """
    Mutual information of each feature with a discrete target, in nats.

    Equal to sklearn.feature_selection.mutual_info_classif(X, y,
    n_neighbors=n_neighbors, random_state=random_state) with continuous
    features.

    device is where the neighbor searches run: 'cpu' in NumPy, or a PyTorch
    device such as 'cuda', in float64, batched over features. Either way the
    scaling, the noise, digamma and the means run in NumPy on the CPU, so
    the result is the same, bit for bit. X is a NumPy array or array-like.
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
    # Each class's members among the used samples, and its k.
    groups = [(codes[used] == label, min(n_neighbors, counts[label] - 1))
              for label in np.flatnonzero(counts > 1)]
    # digamma of every count a sample can have, 1 to all used samples.
    table = digamma(np.arange(1, used.sum() + 1, dtype=np.float64))

    batches = (_counts_numpy(X, used, groups) if device == 'cpu'
               else _counts_torch(X, used, groups, device))
    mi = np.empty(X.shape[1])
    for g0, m in batches:
        # m is features x samples. Each feature's mean over a contiguous
        # row, summed pairwise as NumPy sums the one-dimensional array
        # scikit-learn averages.
        mi[g0:g0 + len(m)] = np.maximum(0, constant - table[m - 1].mean(axis=1))
    return mi


def _counts_numpy(X, used, groups):
    """(first feature, counts m as features x samples) per batch, in NumPy."""
    for g0 in range(0, X.shape[1], _BATCH):
        block = X[used, g0:g0 + _BATCH]
        radius = np.empty_like(block)
        for members, k in groups:
            radius[members] = _kth_neighbor_distance(block[members], k)
        yield g0, np.ascontiguousarray(_counts_within(block, np.nextafter(radius, 0)).T)


def _counts_torch(X, used, groups, device):
    """(first feature, counts m as features x samples) per batch, on a PyTorch device."""
    torch = gpu.import_torch(device)
    members_t = [(torch.as_tensor(np.flatnonzero(members), device=device), k)
                 for members, k in groups]
    zero = torch.zeros((), dtype=torch.float64, device=device)
    size = gpu.batch_size(torch, device, _GPU_BYTES_PER_VALUE * used.sum(), X.shape[1])
    for g0 in range(0, X.shape[1], size):
        # Features as rows, so each feature's samples are contiguous.
        block = torch.as_tensor(X[used, g0:g0 + size], device=device).T.contiguous()
        radius = torch.empty_like(block)
        for members, k in members_t:
            radius[:, members] = _kth_neighbor_distance_torch(torch, block[:, members], k)
        m = _counts_within_torch(torch, block, torch.nextafter(radius, zero))
        yield g0, m.to(torch.int32).cpu().numpy()


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
