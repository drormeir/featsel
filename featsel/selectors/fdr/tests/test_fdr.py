"""
Tests for the fdr selector.
"""

import numpy as np
import pytest
from sklearn.feature_selection import SelectFdr, f_classif

from featsel.selectors import FDRSelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return FDRSelector(alpha=0.05), SelectFdr(f_classif, alpha=0.05)


class TestFDRSelector:
    """Tests for FDRSelector."""

    def test_smaller_alpha_keeps_fewer(self, small_classification_data):
        """Test that a stricter rate never keeps more features."""
        X, y = small_classification_data

        loose = FDRSelector(alpha=0.1).fit(X, y)
        strict = FDRSelector(alpha=0.001).fit(X, y)

        assert 0 < strict.get_support().sum() <= loose.get_support().sum()

    def test_raises_when_nothing_passes(self):
        """Test that pure noise at a tiny alpha fails with a clear message."""
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(100, 20)), rng.integers(0, 2, 100)

        with pytest.raises(ValueError, match="Increase alpha"):
            FDRSelector(alpha=1e-6).fit(X, y)

    def test_regression_task(self, small_regression_data):
        """Test that task='regression' ranks by f_regression."""
        X, y = small_regression_data

        selector = FDRSelector(task='regression').fit(X, y)

        assert selector.get_support().sum() > 0

    def test_requires_target(self, small_classification_data):
        """Test that FDR raises without a target."""
        X, _ = small_classification_data

        with pytest.raises(ValueError, match="requires target"):
            FDRSelector().fit(X, None)

    def test_matches_sklearn(self, sklearn_reference_data):
        """Test that it selects the same features as SelectFdr(f_classif)."""
        assert_same_selection(*_with_sklearn(), *sklearn_reference_data)

    def test_not_slower_than_sklearn(self, sklearn_timing_data):
        """Test that fit is at most 5% slower than SelectFdr(f_classif)."""
        assert_not_slower(*_with_sklearn(), *sklearn_timing_data)
