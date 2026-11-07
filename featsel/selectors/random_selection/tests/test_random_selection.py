"""
Tests for the random_selection selector.
"""

import numpy as np
import pandas as pd
import pytest

from featsel.selectors import RandomSelector


class TestRandomSelector:
    """Tests for RandomSelector, the control baseline."""

    def test_select_exact_count(self, small_classification_data):
        """Test that exactly n_features are selected."""
        X, y = small_classification_data

        X_selected = RandomSelector(n_features=5, random_state=42).fit_transform(X, y)

        assert X_selected.shape[1] == 5
        assert X_selected.shape[0] == X.shape[0]

    def test_reproducibility(self, small_classification_data):
        """Test that the same seed selects the same features."""
        X, y = small_classification_data

        first = RandomSelector(n_features=5, random_state=42).fit(X, y)
        second = RandomSelector(n_features=5, random_state=42).fit(X, y)

        assert np.array_equal(first.get_support(indices=True),
                              second.get_support(indices=True))

    def test_different_seeds_differ(self, small_classification_data):
        """Test that different seeds select different features."""
        X, y = small_classification_data

        first = RandomSelector(n_features=5, random_state=1).fit(X, y)
        second = RandomSelector(n_features=5, random_state=2).fit(X, y)

        assert not np.array_equal(first.get_support(indices=True),
                                  second.get_support(indices=True))

    def test_ignores_target(self, small_classification_data):
        """Test that selection is unchanged when the target is shuffled."""
        X, y = small_classification_data
        y_shuffled = pd.Series(np.random.RandomState(0).permutation(y.values), index=y.index)

        with_y = RandomSelector(n_features=5, random_state=42).fit(X, y)
        with_shuffled = RandomSelector(n_features=5, random_state=42).fit(X, y_shuffled)

        assert np.array_equal(with_y.get_support(indices=True),
                              with_shuffled.get_support(indices=True))

    def test_caps_at_available_features(self, small_classification_data):
        """Test that requesting more features than exist selects all of them."""
        X, y = small_classification_data

        selector = RandomSelector(n_features=X.shape[1] + 10, random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[1] == X.shape[1]

    def test_gpu_tensor_stays_on_gpu(self, small_classification_data, monkeypatch):
        """Test that a GPU tensor is not copied to the CPU and selects as NumPy."""
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('no CUDA GPU')
        X, y = small_classification_data
        X_gpu = torch.tensor(np.asarray(X), device='cuda')
        monkeypatch.setattr(torch.Tensor, 'cpu', lambda self: pytest.fail('copied to the CPU'))

        on_gpu = RandomSelector(n_features=5, random_state=42).fit(X_gpu, y)
        on_cpu = RandomSelector(n_features=5, random_state=42).fit(np.asarray(X), y)

        np.testing.assert_array_equal(on_gpu.get_support(), on_cpu.get_support())
