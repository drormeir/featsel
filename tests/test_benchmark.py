"""Tests for featsel.benchmark."""

import numpy as np

from featsel import benchmark
from featsel.selectors import ANOVAFSelector


def test_overlap_is_against_the_original_only(monkeypatch):
    """Test that past the original's largest size the overlap is NaN, not another variant's."""
    X, y = benchmark.synthetic(n_samples=200, n_genes=300)

    def original():
        return ANOVAFSelector(n_features=5)

    def other():
        return ANOVAFSelector(n_features=5)

    monkeypatch.setattr(benchmark, 'variants', lambda name, gpu: {'original': original,
                                                                   'other': other})
    real_fits = benchmark._fits

    def fits(build, X, y, repeats, max_seconds):
        seconds, supports, device = real_fits(build, X, y, repeats, max_seconds)
        # The original is "too slow" from its first size on, so it stops there.
        return ([max_seconds + 1] if build is original else seconds), supports, device

    monkeypatch.setattr(benchmark, '_fits', fits)
    table = benchmark.benchmark('ANOVAFSelector', X, y, n_genes=(100, 200, 300))

    assert table[table.variant == 'original'].n_genes.tolist() == [100]
    other_rows = table[table.variant == 'other'].set_index('n_genes').jaccard_vs_original
    assert other_rows[100] == 1
    assert np.isnan(other_rows[200]) and np.isnan(other_rows[300])
