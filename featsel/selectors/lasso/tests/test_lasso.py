"""
Tests for the lasso selector.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from featsel.selectors import LassoSelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn(n_features=20, C=0.1):
    """
    This selector and its scikit-learn equivalent, both unfitted.

    The threshold, the smallest positive float, keeps only non-zero
    coefficients, as LassoSelector does.
    """
    return (
        LassoSelector(n_features=n_features, C=C, random_state=0),
        SelectFromModel(LogisticRegression(l1_ratio=1, C=C, solver='liblinear',
                                           max_iter=1000, random_state=0),
                        max_features=n_features, threshold=np.finfo(float).tiny),
    )


class TestLassoSelector:
    """Tests for LassoSelector."""

    def test_lasso_select_top_features(self, small_classification_data):
        """Test selecting a fixed number of features by L1 coefficient."""
        X, y = small_classification_data

        X_selected = LassoSelector(n_features=10, random_state=42).fit_transform(X, y)

        assert X_selected.shape[1] == 10

    def test_lasso_chooses_own_count(self, small_classification_data):
        """Test that without n_features, Lasso keeps only non-zero coefficients."""
        X, y = small_classification_data

        selector = LassoSelector(C=0.1, random_state=42).fit(X, y)

        assert 0 < selector.get_support().sum() <= X.shape[1]

    def test_lasso_sparser_with_smaller_C(self, small_classification_data):
        """Test that stronger regularization keeps fewer features."""
        X, y = small_classification_data

        weak = LassoSelector(C=1.0, random_state=42).fit(X, y)
        strong = LassoSelector(C=0.05, random_state=42).fit(X, y)

        assert strong.get_support().sum() <= weak.get_support().sum()

    def test_lasso_requires_target(self, small_classification_data):
        """Test that Lasso raises without a target."""
        X, _ = small_classification_data

        with pytest.raises(ValueError, match="requires target"):
            LassoSelector(n_features=10).fit(X, None)

    def test_matches_sklearn(self, sklearn_reference_data):
        """Test that it selects the same features as SelectFromModel with L1 logistic regression."""
        assert_same_selection(*_with_sklearn(), *sklearn_reference_data)

    def test_never_selects_a_zero_coefficient(self, sklearn_reference_data):
        """Test that with fewer non-zero coefficients than n_features, only those are kept."""
        X, y = sklearn_reference_data
        selector = LassoSelector(n_features=150, C=0.01, random_state=0).fit(X, y)
        nonzero = selector.feature_importances_ != 0

        assert 0 < nonzero.sum() < 150
        np.testing.assert_array_equal(selector.get_support(), nonzero)
        assert_same_selection(*_with_sklearn(n_features=150, C=0.01), X, y)

    def test_not_slower_than_sklearn(self, sklearn_reference_data):
        """Test that fit is at most 5% slower than SelectFromModel with L1 logistic regression."""
        assert_not_slower(*_with_sklearn(), *sklearn_reference_data)


def _cuda():
    """PyTorch, or skip the test when no CUDA GPU is present."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA GPU')
    return torch


def _converged_liblinear(X, y, C):
    """liblinear's L1 logistic regression run to a tight tolerance: its coef_."""
    return LogisticRegression(l1_ratio=1, C=C, solver='liblinear', tol=1e-10,
                              max_iter=1_000_000).fit(X, y).coef_


class TestPPDNA:
    """LassoSelector with solver='ppdna'."""

    @pytest.mark.parametrize('n_samples, n_features, C, positive', [
        (100, 300, 0.1, 0.5),     # fewer samples than features, as on SCAN-B
        (100, 300, 1.0, 0.5),
        (100, 300, 10.0, 0.5),
        (300, 40, 1.0, 0.5),      # more samples than features
        (200, 300, 1.0, 0.2),     # imbalanced, as most PAM50 subtypes against the rest
        (200, 300, 1.0, 0.1),
        (400, 300, 1.0, 0.05),
    ])
    def test_coefficients_match_converged_liblinear(self, n_samples, n_features, C, positive):
        """Test the coefficients against liblinear run to a tight tolerance."""
        pytest.importorskip('torch')
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=10, weights=[1 - positive], random_state=0)
        expected = _converged_liblinear(X, y, C)

        ours = LassoSelector(C=C, solver='ppdna').fit(X, y).coef_

        np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-6 * np.abs(expected).max())

    def test_selects_as_converged_liblinear(self, sklearn_reference_data):
        """Test that it selects what SelectFromModel does with a converged liblinear."""
        pytest.importorskip('torch')
        reference = SelectFromModel(
            LogisticRegression(l1_ratio=1, C=0.1, solver='liblinear', tol=1e-10,
                               max_iter=1_000_000),
            max_features=20, threshold=np.finfo(float).tiny)
        assert_same_selection(LassoSelector(n_features=20, C=0.1, solver='ppdna'),
                              reference, *sklearn_reference_data)

    def test_regression_raises(self, small_classification_data):
        """Test that PPDNA refuses regression."""
        with pytest.raises(ValueError, match='classification only'):
            LassoSelector(solver='ppdna', task='regression').fit(*small_classification_data)

    def test_liblinear_on_a_gpu_raises(self, small_classification_data):
        """Test that liblinear refuses a device it cannot use."""
        with pytest.raises(ValueError, match='CPU only'):
            LassoSelector(device='cuda').fit(*small_classification_data)

    def test_unknown_solver_raises(self, small_classification_data):
        """Test that a misspelled solver raises."""
        with pytest.raises(ValueError, match='solver must be'):
            LassoSelector(solver='newton').fit(*small_classification_data)

    def test_cuda_selects_as_cpu(self, sklearn_reference_data):
        """Test that the GPU gives the CPU's selection and nearly its coefficients."""
        _cuda()
        X, y = sklearn_reference_data
        cpu = LassoSelector(n_features=20, C=0.1, solver='ppdna').fit(X, y)
        cuda = LassoSelector(n_features=20, C=0.1, solver='ppdna', device='cuda').fit(X, y)

        np.testing.assert_array_equal(cuda.get_support(), cpu.get_support())
        np.testing.assert_allclose(cuda.coef_, cpu.coef_, rtol=0, atol=1e-8 * np.abs(cpu.coef_).max())
        assert cuda.device_ == 'cuda'

    def test_gpu_tensor_runs_on_its_device(self, sklearn_reference_data):
        """Test that a tensor on the GPU is fit there with the default device."""
        torch = _cuda()
        X, y = sklearn_reference_data
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device='cuda')

        selector = LassoSelector(n_features=20, C=0.1, solver='ppdna').fit(X_t, y)

        assert selector.device_ == str(X_t.device)
