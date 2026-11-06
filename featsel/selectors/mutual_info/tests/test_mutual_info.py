"""
Tests for the mutual_info selector.
"""

from functools import partial

import numpy as np
import pytest
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from featsel.selectors import MutualInfoSelector
from featsel.selectors.mutual_info import knn_1d
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return (
        MutualInfoSelector(n_features=20, random_state=0),
        SelectKBest(partial(mutual_info_classif, n_neighbors=3, random_state=0), k=20),
    )


class TestMutualInfoSelector:
    """Tests for MutualInfoSelector."""

    def test_select_features(self, small_classification_data):
        """Test mutual information feature selection."""
        X, y = small_classification_data

        selector = MutualInfoSelector(n_features=10, random_state=42).fit(X, y)

        assert selector.transform(X).shape == (100, 10)
        assert len(selector.get_support(indices=True)) == 10

    def test_requires_n_features(self, small_classification_data):
        """Test that mutual info requires n_features parameter."""
        X, y = small_classification_data

        with pytest.raises(ValueError, match="requires n_features"):
            MutualInfoSelector().fit(X, y)

    def test_reproducibility(self, small_classification_data):
        """Test that random_state ensures reproducibility."""
        X, y = small_classification_data

        first = MutualInfoSelector(n_features=10, random_state=42).fit(X, y)
        second = MutualInfoSelector(n_features=10, random_state=42).fit(X, y)

        assert np.array_equal(first.get_support(indices=True),
                              second.get_support(indices=True))

    def test_matches_sklearn(self, sklearn_reference_data):
        """Test that it selects the same features as SelectKBest(mutual_info_classif)."""
        assert_same_selection(*_with_sklearn(), *sklearn_reference_data)

    def test_not_slower_than_sklearn(self, sklearn_reference_data):
        """Test that fit is at most 5% slower than SelectKBest(mutual_info_classif)."""
        assert_not_slower(*_with_sklearn(), *sklearn_reference_data)


class TestSortedEstimate:
    """knn_1d.mutual_info_classif equals scikit-learn's, bit for bit."""

    @pytest.mark.parametrize('n_samples, n_classes, rounded', [
        (300, 2, False), (300, 3, True), (200, 2, True), (120, 4, False),
    ])
    def test_identical_to_sklearn(self, n_samples, n_classes, rounded):
        """Test equality with continuous and with heavily tied, rounded values."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n_samples, 40)).astype(np.float32)
        if rounded:
            X = np.round(X, 1)
        y = rng.integers(0, n_classes, n_samples)
        X[y == 1, :5] += 1

        np.testing.assert_array_equal(knn_1d.mutual_info_classif(X, y, 3, 42),
                                      mutual_info_classif(X, y, n_neighbors=3,
                                                          random_state=42))

    def test_small_classes_go_to_sklearn(self):
        """Test that a class of at most 7 samples is left to scikit-learn."""
        assert knn_1d.matches_sklearn(np.repeat([0, 1], [8, 50]), 3)
        assert not knn_1d.matches_sklearn(np.repeat([0, 1], [7, 50]), 3)
        assert knn_1d.matches_sklearn(np.repeat([0, 1, 2], [1, 50, 60]), 3)
