"""
Tests for the correlation selector.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from featsel.selectors import CorrelationSelector, correlation, gpu


class TestCorrelationSelector:
    """Tests for CorrelationSelector."""

    def test_select_correlated_features(self, small_regression_data):
        """Test correlation-based selection."""
        X, y = small_regression_data

        selector = CorrelationSelector(n_features=10, target_threshold=0.0).fit(X, y)

        assert selector.transform(X).shape[1] == 10
        assert len(selector.get_support(indices=True)) == 10

    def test_remove_redundant_features(self):
        """Test that highly correlated features are removed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        # Create redundant feature (copy of feature 0)
        X_redundant = np.column_stack([X, X[:, 0] + np.random.randn(100) * 0.01])

        X_df = pd.DataFrame(X_redundant, columns=[f'f{i}' for i in range(6)])
        y = X[:, 0] + X[:, 1]

        selector = CorrelationSelector(inter_feature_threshold=0.9).fit(X_df, y)

        # Should remove one of the redundant features
        assert selector.get_support().sum() < 6

    def test_text_labels_code_as_zero_and_one(self):
        """Test that two text classes select as their 0/1 codes do."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(80, 20)))
        codes = (X[3] + rng.normal(size=80) * 0.5 > 0).astype(int)
        labels = np.where(codes == 1, 'Basal', 'rest')

        by_text = CorrelationSelector(n_features=5, method='spearman').fit(X, labels)
        by_code = CorrelationSelector(n_features=5, method='spearman').fit(X, codes)

        assert list(by_text.get_support(indices=True)) == list(by_code.get_support(indices=True))

    def test_three_text_classes_raise(self):
        """Test that more than two text classes raise: they have no numeric order."""
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 4)))
        with pytest.raises(ValueError, match='exactly two classes'):
            CorrelationSelector().fit(X, np.array(['a', 'b', 'c'] * 10))


def _pandas_pairs(X, threshold, method):
    """The redundant pairs as the selector found them before: the full pandas matrix."""
    corr = X.corr(method=method).abs().to_numpy()
    return set(zip(*np.nonzero(np.triu(corr > threshold, k=1))))


class TestBlockwisePairs:
    """The blockwise redundancy step finds the pairs the full pandas matrix does."""

    @pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
    def test_same_pairs_as_pandas(self, method, monkeypatch):
        """Test on correlated groups, with blocks smaller than the feature count."""
        monkeypatch.setattr(correlation, '_BLOCK', 7)
        rng = np.random.default_rng(0)
        base = rng.normal(size=(150, 6))
        X = pd.DataFrame(np.repeat(base, 5, axis=1) + rng.normal(size=(150, 30)) * 0.25)
        selector = CorrelationSelector(inter_feature_threshold=0.7, method=method)

        ours = {(int(i), int(j)) for i, j in selector._redundant_pairs(X)}

        assert ours == _pandas_pairs(X, 0.7, method)
        assert ours   # the groups do give pairs above the threshold


def _tied_groups():
    """Correlated groups, rounded so Spearman meets ties, and a binary target."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 8))
    X = np.round(np.repeat(base, 4, axis=1) + rng.normal(size=(200, 32)) * 0.3, 1)
    y = (base[:, 0] + base[:, 1] + rng.normal(size=200) > 0).astype(int)
    return X, y


class TestDevice:
    """Pearson and Spearman select the same features on every device."""

    def test_average_ranks_as_scipy(self):
        """Test the PyTorch ranks, with ties, against scipy's average ranks."""
        torch = pytest.importorskip('torch')
        X, _ = _tied_groups()

        ranks = correlation._average_ranks(torch, torch.as_tensor(X))

        np.testing.assert_array_equal(ranks.numpy(), rankdata(X, axis=0))

    @pytest.mark.parametrize('method', ['pearson', 'spearman'])
    def test_torch_path_selects_as_numpy(self, method, monkeypatch):
        """Test the PyTorch path, run on the CPU, in small batches and blocks."""
        pytest.importorskip('torch')
        monkeypatch.setattr(gpu, 'batch_size', lambda *args: 5)
        monkeypatch.setattr(correlation, '_BLOCK', 7)
        X, y = _tied_groups()
        selector = CorrelationSelector(target_threshold=0.05, inter_feature_threshold=0.8,
                                       method=method)

        target_corr, candidates, pairs = selector._on_device(X, y, 'cpu')
        target_ref, candidates_ref, pairs_ref = selector._on_cpu(X, y, list(range(32)))

        np.testing.assert_allclose(target_corr, target_ref, rtol=1e-12)
        np.testing.assert_array_equal(candidates, candidates_ref)
        assert len(pairs) > 0
        assert {(int(i), int(j)) for i, j in pairs} == {(int(i), int(j)) for i, j in pairs_ref}

    @pytest.mark.parametrize('method', ['pearson', 'spearman'])
    def test_gpu_selects_as_cpu(self, method):
        """Test device='cuda', and a GPU tensor with device='cpu', against the CPU."""
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('no CUDA GPU')
        X, y = _tied_groups()
        params = {'n_features': 6, 'target_threshold': 0.05, 'inter_feature_threshold': 0.8,
                  'method': method}
        cpu = CorrelationSelector(device='cpu', **params).fit(X, y)

        cuda = CorrelationSelector(device='cuda', **params).fit(X, y)
        tensor = CorrelationSelector(device='cpu', **params).fit(
            torch.as_tensor(X, device='cuda'), y)

        assert cuda.device_ == 'cuda'
        assert tensor.device_.startswith('cuda')
        np.testing.assert_array_equal(cuda.get_support(), cpu.get_support())
        np.testing.assert_array_equal(tensor.get_support(), cpu.get_support())

    def test_kendall_refuses_a_gpu(self):
        """Test that Kendall raises with a GPU device and runs on the CPU with auto."""
        X, y = _tied_groups()

        with pytest.raises(ValueError, match="kendall' runs on the CPU only"):
            CorrelationSelector(method='kendall', device='cuda').fit(X, y)
        assert CorrelationSelector(method='kendall', target_threshold=0.05).fit(X, y).device_ \
            == 'cpu'
