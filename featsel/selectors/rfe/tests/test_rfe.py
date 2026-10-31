"""
Tests for the rfe selector.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

from featsel.selectors import RFESelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _sklearn(n_features=10, step=0.1, C=1.0):
    """scikit-learn's RFE with a linear SVM, unfitted."""
    return RFE(SVC(kernel='linear', C=C), n_features_to_select=n_features, step=step)


@pytest.fixture(scope='module')
def narrow_data():
    """
    A narrow dataset for comparing selections with scikit-learn.

    Its RFE refits libsvm on every feature each round, so 50 features keeps
    the many reference fits quick.
    """
    return make_classification(n_samples=100, n_features=50, n_informative=10,
                               random_state=0)


class TestRFESelector:
    """Tests for RFESelector."""

    def test_select_top_features(self, small_classification_data):
        """Test that exactly n_features survive the elimination."""
        X, y = small_classification_data

        selector = RFESelector(n_features=5).fit(X, y)

        assert selector.transform(X).shape == (100, 5)

    def test_ranking_marks_kept_features_as_one(self, small_classification_data):
        """Test that ranking_ is 1 exactly where a feature was kept."""
        X, y = small_classification_data

        selector = RFESelector(n_features=5).fit(X, y)

        assert np.array_equal(selector.ranking_ == 1, selector.get_support())

    def test_requires_n_features(self, small_classification_data):
        """Test that RFE needs a target count."""
        X, y = small_classification_data

        with pytest.raises(ValueError, match="requires n_features"):
            RFESelector().fit(X, y)

    def test_needs_two_classes(self, narrow_data):
        """Test that a three-class target is rejected."""
        X, _ = narrow_data

        with pytest.raises(ValueError, match="two classes"):
            RFESelector(n_features=5).fit(X, np.arange(len(X)) % 3)

    @pytest.mark.parametrize('step', [0.1, 0.5, 7])
    def test_matches_sklearn(self, narrow_data, step):
        """Test that it selects the same features as scikit-learn's RFE with a linear SVM."""
        assert_same_selection(RFESelector(n_features=10, step=step), _sklearn(step=step),
                              *narrow_data)

    @pytest.mark.parametrize('step', [0.1, 0.5, 7])
    def test_one_fit_serves_every_larger_k(self, narrow_data, step):
        """Test that the top k of one fit to k=5 is scikit-learn's RFE at each k."""
        X, y = narrow_data
        importances = RFESelector(n_features=5, step=step).fit(X, y).feature_importances_

        for k in [5, 10, 20, 35, 50]:
            top = np.sort(np.argsort(importances)[::-1][:k])
            reference = _sklearn(n_features=k, step=step).fit(X, y).get_support(indices=True)
            np.testing.assert_array_equal(top, reference, err_msg=f'k={k}')

    def test_not_slower_than_sklearn(self):
        """
        Test that fit is at most 5% slower than scikit-learn's RFE with a linear SVM.

        Timed with far fewer samples than features, the case it is built for.
        With more samples than features the Gram matrix costs more than it saves.
        """
        X, y = make_classification(n_samples=100, n_features=1000, n_informative=10,
                                   random_state=0)
        assert_not_slower(RFESelector(n_features=10), _sklearn(), X, y)
