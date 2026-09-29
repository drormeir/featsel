"""
Tests for the variance_threshold selector.
"""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold as SKLearnVarianceThreshold

from featsel.selectors import VarianceThreshold
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return VarianceThreshold(threshold=0.5), SKLearnVarianceThreshold(threshold=0.5)


class TestVarianceThreshold:
    """Tests for VarianceThreshold selector."""

    def test_remove_constant_features(self, data_with_constant_features):
        """Test that constant features are removed."""
        X, _ = data_with_constant_features

        X_selected = VarianceThreshold(threshold=0.0).set_output(transform='pandas').fit(X).transform(X)

        # Should remove 3 constant features
        assert X_selected.shape[1] == 12  # 15 - 3 = 12
        assert not any('constant' in name for name in X_selected.columns)

    def test_remove_low_variance(self, data_with_constant_features):
        """Test that low-variance features are removed."""
        X, _ = data_with_constant_features

        X_selected = VarianceThreshold(threshold=0.1).set_output(transform='pandas').fit(X).transform(X)

        # Should remove 3 constant + 2 low-variance features
        assert X_selected.shape[1] == 10
        assert not any('constant' in name for name in X_selected.columns)
        assert not any('low_var' in name for name in X_selected.columns)

    def test_dataframe_preservation(self, small_classification_data):
        """Test that pandas output keeps a DataFrame, as in scikit-learn."""
        X, _ = small_classification_data

        X_selected = VarianceThreshold().set_output(transform='pandas').fit_transform(X)

        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape[0] == X.shape[0]

    def test_matches_sklearn(self, sklearn_reference_data):
        """Test that it selects the same features as scikit-learn's VarianceThreshold."""
        assert_same_selection(*_with_sklearn(), *sklearn_reference_data)

    def test_not_slower_than_sklearn(self, sklearn_reference_data):
        """Test that fit is at most 5% slower than scikit-learn's VarianceThreshold."""
        assert_not_slower(*_with_sklearn(), *sklearn_reference_data)
