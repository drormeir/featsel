"""
Tests for the correlation selector.
"""

import numpy as np
import pandas as pd
import pytest

from featsel.selectors import CorrelationSelector, correlation


class TestCorrelationSelector:
    """Tests for CorrelationSelector."""

    def test_select_correlated_features(self, small_regression_data):
        """Test correlation-based selection."""
        X, y = small_regression_data

        selector = CorrelationSelector(n_features=10, target_threshold=0.0).fit(X, y)

        assert selector.transform(X).shape[1] == 10
        assert len(selector.get_support(indices=True)) == 10

    def test_remove_redundant_features(self):
        """Test that highly correlated features are removed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        # Create redundant feature (copy of feature 0)
        X_redundant = np.column_stack([X, X[:, 0] + np.random.randn(100) * 0.01])

        X_df = pd.DataFrame(X_redundant, columns=[f'f{i}' for i in range(6)])
        y = X[:, 0] + X[:, 1]

        selector = CorrelationSelector(inter_feature_threshold=0.9).fit(X_df, y)

        # Should remove one of the redundant features
        assert selector.get_support().sum() < 6


def _pandas_pairs(X, threshold, method):
    """The redundant pairs as the selector found them before: the full pandas matrix."""
    corr = X.corr(method=method).abs().to_numpy()
    return set(zip(*np.nonzero(np.triu(corr > threshold, k=1))))


class TestBlockwisePairs:
    """The blockwise redundancy step finds the pairs the full pandas matrix does."""

    @pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
    def test_same_pairs_as_pandas(self, method, monkeypatch):
        """Test on correlated groups, with blocks smaller than the feature count."""
        monkeypatch.setattr(correlation, '_BLOCK', 7)
        rng = np.random.default_rng(0)
        base = rng.normal(size=(150, 6))
        X = pd.DataFrame(np.repeat(base, 5, axis=1) + rng.normal(size=(150, 30)) * 0.25)
        selector = CorrelationSelector(inter_feature_threshold=0.7, method=method)

        ours = {(int(i), int(j)) for i, j in selector._redundant_pairs(X)}

        assert ours == _pandas_pairs(X, 0.7, method)
        assert ours   # the groups do give pairs above the threshold
