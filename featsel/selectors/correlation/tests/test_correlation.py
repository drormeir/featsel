"""
Tests for the correlation selector.
"""

import numpy as np
import pandas as pd

from featsel.selectors import CorrelationSelector


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
