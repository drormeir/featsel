"""
Tests for the lasso selector.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from featsel.selectors import LassoSelector
from tests.sklearn_reference import assert_same_selection

# No speed test against scikit-learn: none of its solvers reaches the exact
# L1 solution. Its Newton solvers refuse the L1 penalty, and liblinear and
# saga stop at a tolerance, so they do less work for a different answer.
# Speed was compared on SCAN-B instead (final_report/06_making_it_fast.md).


def _sklearn(n_features=20, C=0.1):
    """
    scikit-learn's SelectFromModel with L1 logistic regression, liblinear run
    to a tight tolerance, unfitted. The threshold, the smallest positive
    float, keeps only non-zero coefficients, as LassoSelector does.
    """
    return SelectFromModel(LogisticRegression(l1_ratio=1, C=C, solver='liblinear', tol=1e-10,
                                              max_iter=1_000_000),
                           max_features=n_features, threshold=np.finfo(float).tiny)


def _cuda():
    """PyTorch, or skip the test when no CUDA GPU is present."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA GPU')
    return torch


class TestLassoSelector:
    """Tests for LassoSelector."""

    def test_lasso_select_top_features(self, small_classification_data):
        """Test selecting a fixed number of features by L1 coefficient."""
        X, y = small_classification_data

        X_selected = LassoSelector(n_features=10).fit_transform(X, y)

        assert X_selected.shape[1] == 10

    def test_lasso_chooses_own_count(self, small_classification_data):
        """Test that without n_features, Lasso keeps only non-zero coefficients."""
        X, y = small_classification_data

        selector = LassoSelector(C=0.1).fit(X, y)

        assert 0 < selector.get_support().sum() <= X.shape[1]

    def test_lasso_sparser_with_smaller_C(self, small_classification_data):
        """Test that stronger regularization keeps fewer features."""
        X, y = small_classification_data

        weak = LassoSelector(C=1.0).fit(X, y)
        strong = LassoSelector(C=0.05).fit(X, y)

        assert strong.get_support().sum() <= weak.get_support().sum()

    def test_lasso_requires_target(self, small_classification_data):
        """Test that Lasso raises without a target."""
        X, _ = small_classification_data

        with pytest.raises(ValueError, match="requires target"):
            LassoSelector(n_features=10).fit(X, None)

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
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=10, weights=[1 - positive], random_state=0)
        expected = _sklearn(C=C).estimator.fit(X, y).coef_

        ours = LassoSelector(C=C).fit(X, y).coef_

        np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-6 * np.abs(expected).max())

    def test_selects_as_converged_sklearn(self, sklearn_reference_data):
        """Test that it selects what SelectFromModel does with a converged liblinear."""
        assert_same_selection(LassoSelector(n_features=20, C=0.1), _sklearn(),
                              *sklearn_reference_data)

    def test_never_selects_a_zero_coefficient(self, sklearn_reference_data):
        """Test that with fewer non-zero coefficients than n_features, only those are kept."""
        X, y = sklearn_reference_data
        selector = LassoSelector(n_features=150, C=0.01).fit(X, y)
        nonzero = np.abs(selector.coef_).sum(axis=0) != 0

        assert 0 < nonzero.sum() < 150
        np.testing.assert_array_equal(selector.get_support(), nonzero)
        assert_same_selection(LassoSelector(n_features=150, C=0.01),
                              _sklearn(n_features=150, C=0.01), X, y)

    def test_cuda_selects_as_cpu(self, sklearn_reference_data):
        """Test that the GPU gives the CPU's selection and nearly its coefficients."""
        _cuda()
        X, y = sklearn_reference_data
        cpu = LassoSelector(n_features=20, C=0.1).fit(X, y)
        cuda = LassoSelector(n_features=20, C=0.1, device='cuda').fit(X, y)

        np.testing.assert_array_equal(cuda.get_support(), cpu.get_support())
        np.testing.assert_allclose(cuda.coef_, cpu.coef_, rtol=0, atol=1e-8 * np.abs(cpu.coef_).max())
        assert cuda.device_ == 'cuda'

    def test_gpu_tensor_runs_on_its_device(self, sklearn_reference_data):
        """Test that a tensor on the GPU is fit there with the default device."""
        torch = _cuda()
        X, y = sklearn_reference_data
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device='cuda')

        selector = LassoSelector(n_features=20, C=0.1).fit(X_t, y)

        assert selector.device_ == str(X_t.device)


CS = [0.01, 0.1, 1.0, 10.0]


class TestChooseC:
    """LassoSelector with Cs: C chosen by inner cross-validation."""

    def test_chooses_as_converged_sklearn(self, sklearn_reference_data):
        """Test the scores and the choice against the same scan with a converged liblinear."""
        X, y = sklearn_reference_data
        X = np.asarray(X)
        scores = np.array([
            [roc_auc_score(y[test], _sklearn(C=C).estimator.fit(X[train], y[train])
                           .decision_function(X[test])) for C in CS]
            for train, test in StratifiedKFold(3).split(X, y)])

        ours = LassoSelector(Cs=CS, cv=3).fit(X, y)

        np.testing.assert_allclose(ours.cv_results_['mean_test_score'], scores.mean(axis=0),
                                   atol=1e-6)
        assert ours.C_ == CS[int(np.argmax(scores.mean(axis=0)))]
        np.testing.assert_array_equal(ours.get_support(),
                                      LassoSelector(C=ours.C_).fit(X, y).get_support())

    def test_a_tie_goes_to_the_smaller_C(self, sklearn_reference_data):
        """Test that equal scores choose the smallest C among them."""
        X, y = sklearn_reference_data

        selector = LassoSelector(Cs=[5.0, 5.0 + 1e-9], cv=3).fit(X, y)

        assert selector.C_ == 5.0

    def test_records_every_fold(self, sklearn_reference_data):
        """Test that cv_results_ holds each C, sorted, and every fold's score."""
        X, y = sklearn_reference_data

        results = LassoSelector(Cs=[1.0, 0.1], cv=3).fit(X, y).cv_results_

        np.testing.assert_array_equal(results['C'], [0.1, 1.0])
        assert all(len(results[f'split{i}_test_score']) == 2 for i in range(3))

    def test_three_classes_raise(self, small_classification_data):
        """Test that the scan refuses a target with more than two classes."""
        X, _ = small_classification_data
        y = np.arange(len(X)) % 3

        with pytest.raises(ValueError, match='two classes'):
            LassoSelector(Cs=CS).fit(X, y)

    def test_fixed_selects_as_the_search(self, sklearn_reference_data):
        """Test that the fixed selector fits with the chosen C and keeps the same features."""
        X, y = sklearn_reference_data
        searched = LassoSelector(Cs=CS, cv=3).fit(X, y)

        fixed = searched.fixed()

        assert searched.searches() and not fixed.searches()
        assert fixed.C == searched.C_
        np.testing.assert_array_equal(fixed.fit(X, y).get_support(), searched.get_support())

    def test_without_Cs_C_is_used(self, small_classification_data):
        """Test that C_ is C when no scan is asked for."""
        assert LassoSelector(C=0.3).fit(*small_classification_data).C_ == 0.3
