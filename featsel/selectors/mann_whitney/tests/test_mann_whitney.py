"""
Tests for the mann_whitney selector.
"""

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from featsel.selectors import MannWhitneySelector


def _two_classes(n_samples=300, n_features=200, n_signal=10, shift=0.6, seed=0):
    """Two classes; the first n_signal features differ in mean by shift."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_samples // 2)
    X = rng.normal(size=(n_samples, n_features))
    X[y == 1, :n_signal] += shift
    return X, y


class TestMannWhitneySelector:
    """Tests for MannWhitneySelector."""

    def test_keeps_the_smallest_scipy_p_values(self):
        """Test that it keeps the n_features scipy's U test ranks first."""
        X, y = _two_classes()
        p = mannwhitneyu(X[y == 1], X[y == 0], method='asymptotic', axis=0).pvalue

        kept = MannWhitneySelector(n_features=10).fit(X, y).get_support(indices=True)

        np.testing.assert_array_equal(np.sort(kept), np.sort(np.argsort(p)[:10]))

    def test_monotone_transform_keeps_the_selection(self):
        """Test that only the order of each feature matters, as for an AUC."""
        X, y = _two_classes()

        raw = MannWhitneySelector(n_features=10).fit(X, y).get_support()
        transformed = MannWhitneySelector(n_features=10).fit(np.exp(X) ** 3, y).get_support()

        np.testing.assert_array_equal(raw, transformed)

    def test_ranks_where_scipy_gives_zero(self):
        """Test that p-values below 1e-308 still order the features."""
        X, y = _two_classes(n_samples=3000, n_signal=5, shift=3.0)
        X[y == 1, 0] += 1.0  # feature 0 strongest of the five
        assert (mannwhitneyu(X[y == 1], X[y == 0], axis=0).pvalue[:5] == 0).all()

        selector = MannWhitneySelector(n_features=1).fit(X, y)

        assert selector.get_support(indices=True).tolist() == [0]

    def test_requires_n_features(self):
        """Test that it refuses to guess a count."""
        X, y = _two_classes()

        with pytest.raises(ValueError, match='requires n_features'):
            MannWhitneySelector().fit(X, y)

    def test_more_than_two_classes_is_rejected(self):
        """Test that the U test refuses a multiclass target."""
        X, _ = _two_classes()
        y = np.arange(len(X)) % 3

        with pytest.raises(ValueError, match='two classes'):
            MannWhitneySelector(n_features=5).fit(X, y)
