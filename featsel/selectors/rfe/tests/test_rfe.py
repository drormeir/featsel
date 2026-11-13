"""
Tests for the rfe selector.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

from featsel.selectors import RFESelector
from featsel.selectors.rfe.linear_svm import Elimination
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _sklearn(n_features=10, step=0.1, C=1.0):
    """scikit-learn's RFE with a linear SVM at its defaults, unfitted."""
    return RFE(LinearSVC(C=C, max_iter=1000, random_state=0),
               n_features_to_select=n_features, step=step)


def _converged(n_features=10, step=0.1):
    """scikit-learn's RFE with LinearSVC run to a tiny tolerance, unfitted."""
    return RFE(LinearSVC(tol=1e-10, max_iter=100_000), n_features_to_select=n_features,
               step=step)


@pytest.fixture(scope='module')
def narrow_data():
    """
    A narrower dataset for the scikit-learn comparison.

    RFE refits an SVM every round, so the shared 200-feature dataset makes the
    timed fits take over ten seconds.
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


def _cuda():
    """PyTorch, or skip the test when no CUDA GPU is present."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA GPU')
    return torch


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

    @pytest.mark.parametrize('step', [0.1, 0.5, 7])
    def test_selects_as_converged_sklearn(self, narrow_data, step):
        """Test that it selects what scikit-learn's RFE does with a converged LinearSVC."""
        assert_same_selection(RFESelector(n_features=10, step=step), _converged(step=step),
                              *narrow_data)

    def test_three_classes(self, narrow_data):
        """Test that a multiclass target sums the squared weights over classes, as scikit-learn."""
        X, _ = narrow_data
        y = np.arange(len(X)) % 3
        assert_same_selection(RFESelector(n_features=10), _converged(), X, y)

    @pytest.mark.parametrize('step', [0.1, 0.5, 7])
    def test_one_fit_serves_every_larger_k(self, narrow_data, step):
        """Test that the top k of one fit to k=5 is the converged RFE at each k."""
        X, y = narrow_data
        ranking = RFESelector(n_features=5, step=step).fit(X, y).ranking()

        for k in [5, 10, 20, 35, 50]:
            reference = _converged(n_features=k, step=step).fit(X, y).get_support(indices=True)
            np.testing.assert_array_equal(np.sort(ranking[:k]), reference, err_msg=f'k={k}')

    def test_not_slower_than_sklearn(self, wide_data):
        """Test that fit is at most 5% slower than scikit-learn's RFE at its defaults."""
        assert_not_slower(RFESelector(n_features=10), _sklearn(), *wide_data)

    @pytest.mark.parametrize('step', [0.1, 0.5, 7])
    def test_cuda_selects_as_cpu(self, narrow_data, step):
        """Test that the products on the GPU give the CPU's selection."""
        _cuda()
        X, y = narrow_data
        cpu = RFESelector(n_features=10, step=step).fit(X, y)
        cuda = RFESelector(n_features=10, step=step, device='cuda').fit(X, y)

        np.testing.assert_array_equal(cuda.elimination_order_, cpu.elimination_order_)
        assert cuda.device_ == 'cuda'

    def test_gpu_tensor_runs_on_its_device(self, narrow_data):
        """Test that a tensor on the GPU is fit there with device='auto'."""
        torch = _cuda()
        X, y = narrow_data
        X_t = torch.as_tensor(X, dtype=torch.float32, device='cuda')

        selector = RFESelector(n_features=10, device='auto').fit(X_t, y)

        assert selector.device_ == str(X_t.device)


class TestNewtonSVM:
    """The Newton solver reaches LinearSVC's minimum."""

    @pytest.mark.parametrize('n_samples, n_features, C, n_classes', [
        (100, 300, 0.01, 2),    # fewer samples than features, as on SCAN-B
        (100, 300, 1.0, 2),
        (300, 40, 1.0, 2),      # more samples than features
        (150, 200, 0.1, 3),     # one problem per class
    ])
    def test_weights_match_converged_liblinear(self, n_samples, n_features, C, n_classes):
        """Test the squared weights against LinearSVC run to a tiny tolerance."""
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=10, n_classes=n_classes, random_state=0)
        svm = LinearSVC(C=C, tol=1e-10, max_iter=100_000).fit(X, y)
        expected = (svm.coef_ ** 2).sum(axis=0)

        ours = Elimination(X, y, np.unique(y), C, 1000).squared_weights(np.arange(n_features))

        np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-7 * expected.max())
