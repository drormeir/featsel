"""
Tests for the mutual_info selector.
"""

from functools import partial

import numpy as np
import pytest
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from featsel.selectors import MutualInfoSelector
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
