"""
Tests for the percentile selector.
"""

import numpy as np
import pytest
from sklearn.feature_selection import SelectPercentile, f_classif

from featsel.selectors import ANOVAFSelector, PercentileSelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return PercentileSelector(percentile=10), SelectPercentile(f_classif, percentile=10)


class TestPercentileSelector:
    """Tests for PercentileSelector."""

    def test_keeps_the_percentage(self, small_classification_data):
        """Test that 25% of 20 features keeps 5."""
        X, y = small_classification_data

        selector = PercentileSelector(percentile=25).fit(X, y)

        assert selector.transform(X).shape == (100, 5)

    def test_same_ranking_as_anova(self, small_classification_data):
        """Test that it keeps the features ANOVAFSelector ranks highest."""
        X, y = small_classification_data

        percentile = PercentileSelector(percentile=25).fit(X, y).get_support()
        anova = ANOVAFSelector(n_features=5).fit(X, y).get_support()

        assert np.array_equal(percentile, anova)

    def test_regression_task(self, small_regression_data):
        """Test that task='regression' ranks by f_regression."""
        X, y = small_regression_data

        assert PercentileSelector(percentile=50, task='regression').fit(X, y).get_support().sum() == 10

    def test_requires_target(self, small_classification_data):
        """Test that the percentile selector raises without a target."""
        X, _ = small_classification_data

        with pytest.raises(ValueError, match="requires target"):
            PercentileSelector().fit(X, None)

    def test_matches_sklearn(self, sklearn_reference_data):
        """Test that it selects the same features as SelectPercentile(f_classif)."""
        assert_same_selection(*_with_sklearn(), *sklearn_reference_data)

    def test_not_slower_than_sklearn(self, sklearn_timing_data):
        """Test that fit is at most 5% slower than SelectPercentile(f_classif)."""
        assert_not_slower(*_with_sklearn(), *sklearn_timing_data)
