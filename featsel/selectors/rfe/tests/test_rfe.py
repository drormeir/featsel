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


def _sklearn(n_features=10, step=0.1, C=1.0):
    """scikit-learn's RFE with a linear SVM, unfitted."""
    return RFE(LinearSVC(C=C, max_iter=1000, random_state=0),
               n_features_to_select=n_features, step=step)


@pytest.fixture(scope='module')
def narrow_data():
    """
    A narrower dataset for the scikit-learn comparison.

    RFE refits an SVM every round, so the shared 200-feature dataset makes the
    timed fits take over ten seconds.
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

    @pytest.mark.parametrize('step', [0.1, 0.5, 7])
    def test_matches_sklearn(self, narrow_data, step):
        """Test that it selects the same features as scikit-learn's RFE with a linear SVM."""
        ours = RFESelector(n_features=10, step=step, random_state=0)
        assert_same_selection(ours, _sklearn(step=step), *narrow_data)

    def test_matches_sklearn_on_three_classes(self, narrow_data):
        """Test that a multiclass target sums the squared weights over classes, as scikit-learn."""
        X, _ = narrow_data
        y = np.arange(len(X)) % 3
        assert_same_selection(RFESelector(n_features=10, random_state=0), _sklearn(), X, y)

    @pytest.mark.parametrize('step', [0.1, 0.5, 7])
    def test_one_fit_serves_every_larger_k(self, narrow_data, step):
        """Test that the top k of one fit to k=5 is scikit-learn's RFE at each k."""
        X, y = narrow_data
        selector = RFESelector(n_features=5, step=step, random_state=0).fit(X, y)
        importances = selector.feature_importances_

        for k in [5, 10, 20, 35, 50]:
            top = np.sort(np.argsort(importances)[::-1][:k])
            reference = _sklearn(n_features=k, step=step).fit(X, y).get_support(indices=True)
            np.testing.assert_array_equal(top, reference, err_msg=f'k={k}')

    def test_not_slower_than_sklearn(self, narrow_data):
        """Test that fit is at most 5% slower than scikit-learn's RFE with a linear SVM."""
        assert_not_slower(RFESelector(n_features=10, random_state=0), _sklearn(), *narrow_data)
