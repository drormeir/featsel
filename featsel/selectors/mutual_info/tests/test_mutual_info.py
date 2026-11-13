"""
Tests for the mutual_info selector.
"""

from functools import partial

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from featsel.selectors import MutualInfoSelector
from featsel.selectors.mutual_info import knn_1d
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

    def test_one_fit_serves_every_larger_k(self, sklearn_reference_data):
        """Test that the top k of one fit is a separate fit's selection at each k."""
        X, y = sklearn_reference_data
        ranking = MutualInfoSelector(n_features=5, random_state=0).fit(X, y).ranking()

        for k in [5, 20, 50, 120, 200]:
            separate = MutualInfoSelector(n_features=k, random_state=0).fit(X, y)
            np.testing.assert_array_equal(np.sort(ranking[:k]),
                                          separate.get_support(indices=True), err_msg=f'k={k}')

    def test_not_slower_than_sklearn(self, sklearn_reference_data):
        """Test that fit is at most 5% slower than SelectKBest(mutual_info_classif)."""
        assert_not_slower(*_with_sklearn(), *sklearn_reference_data)


class TestSortedEstimate:
    """knn_1d.mutual_info_classif equals scikit-learn's, bit for bit."""

    @pytest.mark.parametrize('n_samples, n_classes, rounded', [
        (300, 2, False), (300, 3, True), (200, 2, True), (120, 4, False),
    ])
    def test_identical_to_sklearn(self, n_samples, n_classes, rounded):
        """Test equality with continuous and with heavily tied, rounded values."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n_samples, 40)).astype(np.float32)
        if rounded:
            X = np.round(X, 1)
        y = rng.integers(0, n_classes, n_samples)
        X[y == 1, :5] += 1

        np.testing.assert_array_equal(knn_1d.mutual_info_classif(X, y, 3, 42),
                                      mutual_info_classif(X, y, n_neighbors=3,
                                                          random_state=42))

    def test_small_classes_go_to_sklearn(self):
        """Test that a class of at most 7 samples is left to scikit-learn."""
        assert knn_1d.matches_sklearn(np.repeat([0, 1], [8, 50]), 3)
        assert not knn_1d.matches_sklearn(np.repeat([0, 1], [7, 50]), 3)
        assert knn_1d.matches_sklearn(np.repeat([0, 1, 2], [1, 50, 60]), 3)


def _cuda():
    """PyTorch, or skip the test when no CUDA GPU is present."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA GPU')
    return torch


def _binary(n_samples, positive, rounded=False):
    """More genes than patients, one class a fraction positive of them."""
    X, y = make_classification(n_samples=n_samples, n_features=2 * n_samples, n_informative=10,
                               weights=[1 - positive], random_state=0)
    return (np.round(X, 1) if rounded else X).astype(np.float32), y


class TestGPU:
    """The neighbor searches on a GPU give scikit-learn's result, bit for bit."""

    @pytest.mark.parametrize('n_samples, positive, rounded', [
        (200, 0.5, False), (200, 0.2, True), (300, 0.1, False), (400, 0.05, True),
    ])
    def test_identical_to_sklearn(self, n_samples, positive, rounded):
        """Test equality with continuous and with heavily tied, rounded values."""
        _cuda()
        X, y = _binary(n_samples, positive, rounded)

        np.testing.assert_array_equal(
            knn_1d.mutual_info_classif(X, y, 3, 42, device='cuda'),
            mutual_info_classif(X, y, n_neighbors=3, random_state=42))

    @pytest.mark.parametrize('layout', ['column-major', 'one gene'])
    def test_identical_in_numpy_sum_order(self, layout):
        """Test the layouts in which NumPy sums each gene pairwise, not sample by sample."""
        _cuda()
        X, y = _binary(200, 0.2)
        X = np.asfortranarray(X) if layout == 'column-major' else X[:, :1].copy()

        np.testing.assert_array_equal(
            knn_1d.mutual_info_classif(X, y, 3, 42, device='cuda'),
            mutual_info_classif(X, y, n_neighbors=3, random_state=42))

    def test_cuda_selects_as_cpu(self):
        """Test that the selector gives the CPU's scores on the GPU and says where it ran."""
        _cuda()
        X, y = _binary(300, 0.2)
        cpu = MutualInfoSelector(n_features=20, device='cpu').fit(X, y)
        cuda = MutualInfoSelector(n_features=20, device='cuda').fit(X, y)

        np.testing.assert_array_equal(cuda.scores_, cpu.scores_)
        assert (cpu.device_, cuda.device_) == ('cpu', 'cuda')

    def test_gpu_tensor_runs_on_its_device(self):
        """Test that a tensor on the GPU is fit there, even with device='cpu'."""
        torch = _cuda()
        X, y = _binary(300, 0.1)
        X_t = torch.as_tensor(X, device='cuda')

        selector = MutualInfoSelector(n_features=20, device='cpu').fit(X_t, y)

        assert selector.device_ == str(X_t.device)
        np.testing.assert_array_equal(selector.scores_,
                                      mutual_info_classif(X, y, n_neighbors=3, random_state=42))

    def test_small_class_goes_to_sklearn_on_cpu(self):
        """Test that a class of at most 7 samples runs in scikit-learn, from a GPU tensor."""
        torch = _cuda()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 200)).astype(np.float32)
        y = np.repeat([0, 1], [95, 5])

        selector = MutualInfoSelector(n_features=20, device='cuda').fit(
            torch.as_tensor(X, device='cuda'), y)

        assert selector.device_ == 'cpu'
        np.testing.assert_array_equal(selector.scores_,
                                      mutual_info_classif(X, y, n_neighbors=3, random_state=42))
