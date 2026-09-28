"""
Tests for the mutual_info selector.
"""

import numpy as np
import pytest

from featsel.selectors import MutualInfoSelector


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
