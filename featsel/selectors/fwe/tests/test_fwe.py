"""
Tests for the fwe selector.
"""

import numpy as np
import pytest
from sklearn.feature_selection import SelectFwe, f_classif

from featsel.selectors import FDRSelector, FWESelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return FWESelector(alpha=0.05), SelectFwe(f_classif, alpha=0.05)


class TestFWESelector:
    """Tests for FWESelector."""

    def test_keeps_features_below_the_bonferroni_cutoff(self, small_classification_data):
        """Test that exactly the features with p < alpha / n_features are kept."""
        X, y = small_classification_data

        selector = FWESelector(alpha=0.05).fit(X, y)

        assert np.array_equal(selector.get_support(), selector.pvalues_ < 0.05 / X.shape[1])

    def test_stricter_than_fdr(self, small_classification_data):
        """Test that every feature Bonferroni keeps, Benjamini-Hochberg keeps too."""
        X, y = small_classification_data

        fwe = FWESelector(alpha=0.05).fit(X, y).get_support()
        fdr = FDRSelector(alpha=0.05).fit(X, y).get_support()

        assert np.all(fdr[fwe])

    def test_raises_when_nothing_passes(self):
        """Test that pure noise at a tiny alpha fails with a clear message."""
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(100, 20)), rng.integers(0, 2, 100)

        with pytest.raises(ValueError, match="Increase alpha"):
            FWESelector(alpha=1e-6).fit(X, y)

    def test_requires_target(self, small_classification_data):
        """Test that FWE raises without a target."""
        X, _ = small_classification_data

        with pytest.raises(ValueError, match="requires target"):
            FWESelector().fit(X, None)

    def test_matches_sklearn(self, sklearn_reference_data):
        """Test that it selects the same features as SelectFwe(f_classif)."""
        assert_same_selection(*_with_sklearn(), *sklearn_reference_data)

    def test_not_slower_than_sklearn(self, sklearn_timing_data):
        """Test that fit is at most 5% slower than SelectFwe(f_classif)."""
        assert_not_slower(*_with_sklearn(), *sklearn_timing_data)
