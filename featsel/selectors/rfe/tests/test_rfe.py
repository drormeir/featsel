"""
Tests for the rfe selector.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

from featsel.selectors import RFESelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return (
        RFESelector(n_features=10, step=0.1, random_state=0),
        RFE(LinearSVC(C=1.0, max_iter=1000, random_state=0), n_features_to_select=10, step=0.1),
    )


@pytest.fixture(scope='module')
def narrow_data():
    """
    A narrower dataset for the scikit-learn comparison.

    RFE refits an SVM every round, so the shared 200-feature dataset makes the
    20 timed fits take over ten seconds.
    """
    return make_classification(n_samples=100, n_features=50, n_informative=10,
                               random_state=0)


class TestRFESelector:
    """Tests for RFESelector."""

    def test_select_top_features(self, small_classification_data):
        """Test that exactly n_features survive the elimination."""
        X, y = small_classification_data

        selector = RFESelector(n_features=5, random_state=0).fit(X, y)

        assert selector.transform(X).shape == (100, 5)

    def test_ranking_marks_kept_features_as_one(self, small_classification_data):
        """Test that ranking_ is 1 exactly where a feature was kept."""
        X, y = small_classification_data

        selector = RFESelector(n_features=5, random_state=0).fit(X, y)

        assert np.array_equal(selector.ranking_ == 1, selector.get_support())

    def test_requires_n_features(self, small_classification_data):
        """Test that RFE needs a target count."""
        X, y = small_classification_data

        with pytest.raises(ValueError, match="requires n_features"):
            RFESelector().fit(X, y)

    def test_matches_sklearn(self, narrow_data):
        """Test that it selects the same features as scikit-learn's RFE with a linear SVM."""
        assert_same_selection(*_with_sklearn(), *narrow_data)

    def test_not_slower_than_sklearn(self, narrow_data):
        """Test that fit is at most 5% slower than scikit-learn's RFE with a linear SVM."""
        assert_not_slower(*_with_sklearn(), *narrow_data)
