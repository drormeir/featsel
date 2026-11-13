"""
Tests for the rfecv selector.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC

from featsel.selectors import RFECVSelector
from tests.sklearn_reference import assert_not_slower


def _converged(scoring=None):
    """scikit-learn's RFECV with LinearSVC run to a tiny tolerance, unfitted."""
    return RFECV(LinearSVC(tol=1e-10, max_iter=100_000), step=0.1, cv=3, scoring=scoring)


@pytest.fixture(scope='module')
def narrow_data():
    """
    A narrower dataset for the scikit-learn comparison.

    RFECV runs a full elimination in every inner fold, so the shared
    200-feature dataset makes the 20 timed fits take over twenty seconds.
    """
    return make_classification(n_samples=100, n_features=50, n_informative=10,
                               random_state=0)


@pytest.fixture(scope='module')
def wide_data():
    """
    More features than samples, the shape this package is for, to time on.
    The Newton solver works in the samples' space, so with more samples
    than features, as in narrow_data, liblinear is faster.
    """
    return make_classification(n_samples=100, n_features=2000, n_informative=10,
                               random_state=0)


class TestRFECVSelector:
    """Tests for RFECVSelector."""

    def test_keeps_the_chosen_count(self, small_classification_data):
        """Test that the support holds exactly the count cross-validation chose."""
        X, y = small_classification_data

        selector = RFECVSelector(cv=3).fit(X, y)

        assert selector.get_support().sum() == selector.n_features_
        assert np.array_equal(selector.ranking_ == 1, selector.get_support())

    def test_respects_the_minimum(self, small_classification_data):
        """Test that it never keeps fewer than min_features_to_select."""
        X, y = small_classification_data

        selector = RFECVSelector(cv=3, min_features_to_select=8).fit(X, y)

        assert selector.n_features_ >= 8

    def test_records_scores_per_count(self, small_classification_data):
        """Test that cv_results_ holds one mean score per feature count tried."""
        X, y = small_classification_data

        selector = RFECVSelector(cv=3).fit(X, y)

        assert len(selector.cv_results_['mean_test_score']) == len(selector.cv_results_['n_features'])

    @pytest.mark.parametrize('scoring', [None, 'f1_macro'])
    def test_matches_converged_sklearn(self, narrow_data, scoring):
        """Test count, ranking and scores against RFECV with a converged LinearSVC."""
        X, y = narrow_data
        reference = _converged(scoring).fit(X, y)

        ours = RFECVSelector(step=0.1, cv=3, scoring=scoring).fit(X, y)

        assert ours.n_features_ == reference.n_features_
        np.testing.assert_array_equal(ours.ranking_, reference.ranking_)
        np.testing.assert_array_equal(ours.cv_results_['n_features'],
                                      reference.cv_results_['n_features'])
        np.testing.assert_allclose(ours.cv_results_['mean_test_score'],
                                   reference.cv_results_['mean_test_score'])

    def test_three_classes(self, narrow_data):
        """Test a one-vs-rest target against scikit-learn."""
        X, _ = narrow_data
        y = np.arange(len(X)) % 3
        reference = _converged().fit(X, y)

        ours = RFECVSelector(step=0.1, cv=3).fit(X, y)

        assert ours.n_features_ == reference.n_features_
        np.testing.assert_array_equal(ours.support_, reference.support_)

    def test_fixed_is_rfe_at_the_chosen_count(self, narrow_data):
        """Test that the fixed selector, RFE at the chosen count, keeps the same features."""
        searched = RFECVSelector(step=0.1, cv=3).fit(*narrow_data)

        fixed = searched.fixed()

        assert fixed.n_features == searched.n_features_
        np.testing.assert_array_equal(fixed.fit(*narrow_data).get_support(),
                                      searched.get_support())
        np.testing.assert_array_equal(fixed.ranking_, searched.ranking_)

    def test_not_slower_than_sklearn(self, wide_data):
        """Test that fit is at most 5% slower than scikit-learn's RFECV at its defaults."""
        reference = RFECV(LinearSVC(C=1.0, max_iter=1000, random_state=0), step=0.1, cv=3)
        assert_not_slower(RFECVSelector(step=0.1, cv=3), reference, *wide_data)

    def test_cuda_matches_cpu(self, narrow_data):
        """Test that the GPU products give the CPU's choice."""
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('no CUDA GPU')
        cpu = RFECVSelector(cv=3).fit(*narrow_data)
        cuda = RFECVSelector(cv=3, device='cuda').fit(*narrow_data)

        np.testing.assert_array_equal(cuda.ranking_, cpu.ranking_)
        assert cuda.device_ == 'cuda'
