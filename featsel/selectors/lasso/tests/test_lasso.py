"""
Tests for the lasso selector.
"""

import pytest

from featsel.selectors import LassoSelector


class TestLassoSelector:
    """Tests for LassoSelector."""

    def test_lasso_select_top_features(self, small_classification_data):
        """Test selecting a fixed number of features by L1 coefficient."""
        X, y = small_classification_data

        X_selected = LassoSelector(n_features=10, random_state=42).fit_transform(X, y)

        assert X_selected.shape[1] == 10

    def test_lasso_chooses_own_count(self, small_classification_data):
        """Test that without n_features, Lasso keeps only non-zero coefficients."""
        X, y = small_classification_data

        selector = LassoSelector(C=0.1, random_state=42).fit(X, y)

        assert 0 < selector.get_support().sum() <= X.shape[1]

    def test_lasso_sparser_with_smaller_C(self, small_classification_data):
        """Test that stronger regularization keeps fewer features."""
        X, y = small_classification_data

        weak = LassoSelector(C=1.0, random_state=42).fit(X, y)
        strong = LassoSelector(C=0.05, random_state=42).fit(X, y)

        assert strong.get_support().sum() <= weak.get_support().sum()

    def test_lasso_requires_target(self, small_classification_data):
        """Test that Lasso raises without a target."""
        X, _ = small_classification_data

        with pytest.raises(ValueError, match="requires target"):
            LassoSelector(n_features=10).fit(X, None)
