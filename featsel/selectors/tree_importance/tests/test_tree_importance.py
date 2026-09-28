"""
Tests for the tree_importance selector.
"""

import numpy as np

from featsel.selectors import TreeImportanceSelector


class TestTreeImportanceSelector:
    """Tests for TreeImportanceSelector."""

    def test_tree_importance_select_top_features(self, small_classification_data):
        """Test selecting a fixed number of features by impurity decrease."""
        X, y = small_classification_data

        selector = TreeImportanceSelector(n_features=10, n_estimators=20, random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[1] == 10

    def test_tree_importance_reproducibility(self, small_classification_data):
        """Test that the same seed gives the same ranking."""
        X, y = small_classification_data

        params = {'n_features': 10, 'n_estimators': 20, 'random_state': 42}
        first = TreeImportanceSelector(**params).fit(X, y)
        second = TreeImportanceSelector(**params).fit(X, y)

        assert np.array_equal(first.get_support(indices=True),
                              second.get_support(indices=True))
