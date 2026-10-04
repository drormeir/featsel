"""
Tests for the rfecv selector.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC

from featsel.selectors import RFECVSelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return (
        RFECVSelector(step=0.1, cv=3, random_state=0),
        RFECV(LinearSVC(C=1.0, max_iter=1000, random_state=0), step=0.1, cv=3),
    )


@pytest.fixture(scope='module')
def narrow_data():
    """
    A narrower dataset for the scikit-learn comparison.

    RFECV runs a full elimination in every inner fold, so the shared
    200-feature dataset makes the 20 timed fits take over twenty seconds.
    """
    return make_classification(n_samples=100, n_features=50, n_informative=10,
                               random_state=0)


class TestRFECVSelector:
    """Tests for RFECVSelector."""

    def test_keeps_the_chosen_count(self, small_classification_data):
        """Test that the support holds exactly the count cross-validation chose."""
        X, y = small_classification_data

        selector = RFECVSelector(cv=3, random_state=0).fit(X, y)

        assert selector.get_support().sum() == selector.n_features_
        assert np.array_equal(selector.ranking_ == 1, selector.get_support())

    def test_respects_the_minimum(self, small_classification_data):
        """Test that it never keeps fewer than min_features_to_select."""
        X, y = small_classification_data

        selector = RFECVSelector(cv=3, min_features_to_select=8, random_state=0).fit(X, y)

        assert selector.n_features_ >= 8

    def test_records_scores_per_count(self, small_classification_data):
        """Test that cv_results_ holds one mean score per feature count tried."""
        X, y = small_classification_data

        selector = RFECVSelector(cv=3, random_state=0).fit(X, y)

        assert len(selector.cv_results_['mean_test_score']) == len(selector.cv_results_['n_features'])

    def test_matches_sklearn(self, narrow_data):
        """Test that it selects the same features as scikit-learn's RFECV with a linear SVM."""
        assert_same_selection(*_with_sklearn(), *narrow_data)

    def test_not_slower_than_sklearn(self, narrow_data):
        """Test that fit is at most 5% slower than scikit-learn's RFECV with a linear SVM."""
        assert_not_slower(*_with_sklearn(), *narrow_data)
