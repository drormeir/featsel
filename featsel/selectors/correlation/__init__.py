"""
Correlation: rank features by target correlation and drop redundant pairs.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .. import gpu
from ..base import FeatureSelector

# Candidate features compared per block in the redundancy step. A block's
# correlations are _BLOCK x n_candidates float64 values: 80 MB at 10,000.
_BLOCK = 1024

# Peak bytes per value of a GPU batch while standardizing it: the float64
# copy, and for Spearman its sort, order, tie bounds and ranks.
_BYTES_PER_VALUE = 96


class CorrelationSelector(FeatureSelector):
    """
    Select features based on correlation with target and remove redundant features.

    Best for: Removing redundancy, simple linear relationships
    Pros: Simple (O(m^2*n)), interpretable, reduces multicollinearity
    Cons: Only linear correlation, may remove important correlated features

    The redundancy step never holds the full correlation matrix. For
    Pearson and Spearman it standardizes the candidates once and finds the
    pairs above the threshold block by block, with one matrix product per
    block. Kendall goes through pandas, on the CPU only.

    Strategy:
    1. Compute correlation of each feature with target
    2. Keep features above target_threshold
    3. Among highly correlated feature pairs, keep the one with higher target correlation
    4. Select top n_features by target correlation (if specified)

    Parameters
    ----------
    n_features : int, optional
        Number of features to select. If None, uses target_threshold only.
    target_threshold : float, default=0.1
        Minimum absolute correlation with target to keep feature.
    inter_feature_threshold : float, default=0.95
        Maximum correlation between features (remove one if exceeded).
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'.

    device : str, default='auto'
        Where Pearson and Spearman correlations are computed: 'auto' uses a
        CUDA GPU when one is free (the `gpu` extra), else NumPy and pandas;
        'cpu' always those; 'cuda' always the GPU, waiting while another fit
        holds it. A tensor on a GPU runs there whatever this says. Kendall
        runs on the CPU, and raises with a GPU device. The GPU rounds
        differently, so a correlation at a threshold to the last digits can
        fall on the other side.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    device_ : str
        The device the correlations were computed on: 'cpu' or 'cuda'.
    target_corr_ : pd.Series
        Absolute correlation of each feature with target.
    feature_importances_ : np.ndarray
        Alias for target_corr_ values.
    selected_features_ : list
        Names (DataFrame input) or indices of selected features.

    Examples
    --------
    >>> from featsel.selectors import CorrelationSelector
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    >>> y = X['f0'] + X['f1'] + np.random.randn(100) * 0.1
    >>> CorrelationSelector(n_features=3).fit_transform(X, y).shape
    (100, 3)
    """

    def __init__(self, n_features=None, target_threshold=0.1,
                 inter_feature_threshold=0.95, method='pearson', device='auto',
                 check_input=True):
        self.n_features = n_features
        self.target_threshold = target_threshold
        self.inter_feature_threshold = inter_feature_threshold
        self.method = method
        self.device = device
        self.check_input = check_input

    def fit(self, X, y):
        """
        Fit the selector on training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : CorrelationSelector
            Fitted selector.
        """
        X, y = self._validate(X, y, keep_gpu_tensor=True)
        names = list(getattr(self, 'feature_names_in_', range(self.n_features_in_)))
        y = np.asarray(y)
        if not np.issubdtype(y.dtype, np.number):
            # Two class labels code as 0 and 1; more have no numeric order.
            classes, y = np.unique(y, return_inverse=True)
            if len(classes) != 2:
                raise ValueError(
                    f"A non-numeric target needs exactly two classes, got {len(classes)}")

        device = self.device
        if self.method == 'kendall':
            if device not in ('auto', 'cpu'):
                raise ValueError(f"method='kendall' runs on the CPU only, not device={device!r}")
            device = 'cpu'
            if not isinstance(X, np.ndarray):
                X = X.cpu().numpy()       # a GPU tensor

        # 1. Correlation of each feature with the target; 2. the features
        # above target_threshold; 3. the redundant pairs among those.
        (target_corr, candidate_indices, pairs), self.device_ = gpu.run(
            device, X,
            lambda X: self._on_cpu(X, y, names),
            lambda X, device: self._on_device(X, y, device))
        self.target_corr_ = pd.Series(target_corr, index=names)
        candidates = [names[i] for i in candidate_indices]

        # Of each redundant pair, drop the feature with the lower target correlation
        to_drop = set()
        for i, j in pairs:
            # Drop feature with lower target correlation
            feat_i, feat_j = candidates[i], candidates[j]
            if self.target_corr_[feat_i] > self.target_corr_[feat_j]:
                to_drop.add(feat_j)
            else:
                to_drop.add(feat_i)

        selected = [f for f in candidates if f not in to_drop]

        if len(selected) == 0:
            raise ValueError("All features were removed due to inter-feature correlation threshold")

        # 4. Select top n_features by target correlation
        if self.n_features is not None and len(selected) > self.n_features:
            selected = self.target_corr_[selected].nlargest(self.n_features).index.tolist()

        self.selected_features_ = selected
        self.feature_importances_ = self.target_corr_.values
        self.support_ = np.isin(names, selected)
        return self

    def _candidates(self, target_corr):
        """Indices of the features at or above target_threshold; raises if none."""
        candidates = np.flatnonzero(target_corr >= self.target_threshold)
        if len(candidates) == 0:
            raise ValueError(
                f"No features meet target_threshold={self.target_threshold}. "
                f"Max correlation: {np.nanmax(target_corr):.3f}"
            )
        return candidates

    def _on_cpu(self, X, y, names):
        """Target correlations, candidates and redundant pairs, in pandas and NumPy."""
        X = pd.DataFrame(X, columns=names)
        target_corr = X.corrwith(pd.Series(y), method=self.method).abs().to_numpy()
        candidates = self._candidates(target_corr)
        return target_corr, candidates, self._redundant_pairs(X.iloc[:, candidates])

    def _on_device(self, X, y, device):
        """The same as _on_cpu, Pearson or Spearman, in float64 on device."""
        torch = gpu.import_torch(device)
        y = np.asarray(y, dtype=np.float64)
        if self.method == 'spearman':
            y = rankdata(y)
        y = y - y.mean()
        y_t = torch.as_tensor(y / np.linalg.norm(y), device=device)
        target_corr = np.concatenate([
            (y_t @ self._standardized(torch, block)).abs().cpu().numpy()
            for block in gpu.float64_columns(torch, X, device, _BYTES_PER_VALUE)])
        candidates = self._candidates(target_corr)

        chosen = X[:, candidates if isinstance(X, np.ndarray)
                   else torch.as_tensor(candidates, device=X.device)]
        Z = torch.empty(chosen.shape, dtype=torch.float64, device=device)
        start = 0
        for block in gpu.float64_columns(torch, chosen, device, _BYTES_PER_VALUE):
            Z[:, start:start + block.shape[1]] = self._standardized(torch, block)
            start += block.shape[1]
        del chosen
        pairs = []
        for i0 in range(0, Z.shape[1], _BLOCK):
            corr = (Z[:, i0:i0 + _BLOCK].T @ Z[:, i0:]).abs()
            i, j = torch.nonzero(torch.triu(corr > self.inter_feature_threshold, diagonal=1),
                                 as_tuple=True)
            pairs.extend(zip((i + i0).tolist(), (j + i0).tolist()))
        return target_corr, candidates, pairs

    def _standardized(self, torch, block):
        """Each column centered to mean 0 and scaled to norm 1; ranked first for Spearman."""
        if self.method == 'spearman':
            block = _average_ranks(torch, block)
        block = block - block.mean(0)
        return block / torch.linalg.vector_norm(block, dim=0)

    def _redundant_pairs(self, X):
        """Pairs (i, j), i < j, of columns of X correlated above the threshold."""
        if self.method == 'kendall':
            corr = X.corr(method='kendall').abs().to_numpy()
            return zip(*np.nonzero(np.triu(corr > self.inter_feature_threshold, k=1)))
        Z = X.to_numpy(dtype=np.float64)
        if self.method == 'spearman':
            Z = rankdata(Z, axis=0)          # average ranks, as pandas ranks
        Z = Z - Z.mean(axis=0)
        Z /= np.linalg.norm(Z, axis=0)
        pairs = []
        for i0 in range(0, Z.shape[1], _BLOCK):
            # Only columns from i0 on: pairs with i < j.
            corr = np.abs(Z[:, i0:i0 + _BLOCK].T @ Z[:, i0:])
            i, j = np.nonzero(np.triu(corr > self.inter_feature_threshold, k=1))
            pairs.extend(zip(i + i0, j + i0))
        return pairs


def _average_ranks(torch, block):
    """Each column's values replaced by their ranks, ties by their average rank."""
    values, order = torch.sort(block, dim=0)
    n = block.shape[0]
    position = torch.arange(n, device=block.device)[:, None]
    first = torch.ones(values.shape, dtype=torch.bool, device=block.device)
    first[1:] = values[1:] != values[:-1]          # a value's first place among its ties
    last = torch.ones_like(first)
    last[:-1] = first[1:]                          # its last place
    start = torch.cummax(torch.where(first, position, 0), dim=0).values
    end = torch.cummin(torch.where(last, position, n).flip(0), dim=0).values.flip(0)
    ranks = torch.empty_like(values)
    ranks.scatter_(0, order, (start + end).to(values.dtype) / 2 + 1)
    return ranks
