"""
Tests for the variance_threshold selector.
"""

import pandas as pd

from featsel.selectors import VarianceThreshold


class TestVarianceThreshold:
    """Tests for VarianceThreshold selector."""

    def test_remove_constant_features(self, data_with_constant_features):
        """Test that constant features are removed."""
        X, _ = data_with_constant_features

        X_selected = VarianceThreshold(threshold=0.0).fit(X).transform(X)

        # Should remove 3 constant features
        assert X_selected.shape[1] == 12  # 15 - 3 = 12
        assert not any('constant' in name for name in X_selected.columns)

    def test_remove_low_variance(self, data_with_constant_features):
        """Test that low-variance features are removed."""
        X, _ = data_with_constant_features

        X_selected = VarianceThreshold(threshold=0.1).fit(X).transform(X)

        # Should remove 3 constant + 2 low-variance features
        assert X_selected.shape[1] == 10
        assert not any('constant' in name for name in X_selected.columns)
        assert not any('low_var' in name for name in X_selected.columns)

    def test_dataframe_preservation(self, small_classification_data):
        """Test that DataFrame input returns DataFrame output."""
        X, _ = small_classification_data

        X_selected = VarianceThreshold().fit_transform(X)

        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape[0] == X.shape[0]
