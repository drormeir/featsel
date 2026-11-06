"""
Tests for the sequential selector.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

from featsel.selectors import SequentialSelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return (
        SequentialSelector(n_features=3, cv=3),
        SequentialFeatureSelector(LogisticRegression(C=1.0, max_iter=1000),
                                  n_features_to_select=3, cv=3),
    )


@pytest.fixture(scope='module')
def narrow_data():
    """
    A narrow dataset for the scikit-learn comparison.

    Sequential selection fits one model per candidate feature per step, so the
    shared 200-feature dataset would take minutes per fit.
    """
    return make_classification(n_samples=60, n_features=8, n_informative=3,
                               n_redundant=0, random_state=0)


class TestSequentialSelector:
    """Tests for SequentialSelector."""

    def test_forward_keeps_n_features(self, narrow_data):
        """Test that forward selection stops at n_features."""
        X, y = narrow_data

        selector = SequentialSelector(n_features=3, cv=3).fit(X, y)

        assert selector.transform(X).shape == (60, 3)

    def test_backward_keeps_n_features(self, narrow_data):
        """Test that backward selection stops at n_features."""
        X, y = narrow_data

        selector = SequentialSelector(n_features=3, direction='backward', cv=3).fit(X, y)

        assert selector.get_support().sum() == 3

    def test_requires_n_features(self, narrow_data):
        """Test that sequential selection needs a target count."""
        X, y = narrow_data

        with pytest.raises(ValueError, match="requires n_features"):
            SequentialSelector().fit(X, y)

    def test_matches_sklearn(self, narrow_data):
        """Test that it selects the same features as SequentialFeatureSelector."""
        assert_same_selection(*_with_sklearn(), *narrow_data)

    def test_not_slower_than_sklearn(self, narrow_data):
        """Test that fit is at most 5% slower than SequentialFeatureSelector."""
        assert_not_slower(*_with_sklearn(), *narrow_data)


class TestNewtonSequential:
    """SequentialSelector with solver='newton'."""

    @pytest.mark.parametrize('seed', [0, 1, 2])
    @pytest.mark.parametrize('C', [1.0, 0.05])
    def test_selects_as_converged_lbfgs(self, seed, C):
        """Test the selection against scikit-learn with a converged LogisticRegression."""
        pytest.importorskip('torch')
        X, y = make_classification(n_samples=200, n_features=30, n_informative=6,
                                   random_state=seed)
        reference = SequentialFeatureSelector(
            LogisticRegression(C=C, tol=1e-10, max_iter=10_000),
            n_features_to_select=4, cv=5).fit(X, y)

        ours = SequentialSelector(n_features=4, cv=5, C=C, solver='newton').fit(X, y)

        np.testing.assert_array_equal(ours.get_support(), reference.support_)

    def test_cuda_selects_as_cpu(self):
        """Test that the GPU adds the same features in the same order."""
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('no CUDA GPU')
        X, y = make_classification(n_samples=200, n_features=30, n_informative=6,
                                   random_state=0)

        cpu = SequentialSelector(n_features=4, solver='newton').fit(X, y)
        cuda = SequentialSelector(n_features=4, solver='newton', device='cuda').fit(X, y)

        assert cuda.selection_order_ == cpu.selection_order_
        assert cuda.device_ == 'cuda'

    @pytest.mark.parametrize('params, message', [
        ({'direction': 'backward'}, 'forward'),
        ({'scoring': 'f1_macro'}, 'accuracy'),
        ({'device': 'cuda', 'solver': 'lbfgs'}, 'CPU only'),
    ])
    def test_unsupported_settings_raise(self, params, message):
        """Test that the newton solver refuses what it does not implement."""
        X, y = make_classification(n_samples=50, n_features=8, random_state=0)
        settings = {'solver': 'newton', **params}

        with pytest.raises(ValueError, match=message):
            SequentialSelector(n_features=2, **settings).fit(X, y)

    def test_more_than_two_classes_raise(self):
        """Test that the newton solver needs a binary target."""
        X, _ = make_classification(n_samples=60, n_features=8, random_state=0)
        y = np.arange(60) % 3

        with pytest.raises(ValueError, match='two classes'):
            SequentialSelector(n_features=2, solver='newton').fit(X, y)
