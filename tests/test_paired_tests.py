"""
Tests for scripts/paired_tests.py: which scores a baseline is paired with.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import paired_tests


def _results(baseline_k):
    """Four splits of one method at k = 5 and 10, and a baseline at baseline_k."""
    rows = []
    for split in range(4):
        for k, score in [(5, 0.6), (10, 0.8)]:
            rows.append({'selector': 'Method', 'k': k, 'split': split,
                         'score': score + 0.01 * split, 'n_test': 30, 'n_train': 70})
        for k in baseline_k:
            rows.append({'selector': 'Base', 'k': k, 'split': split,
                         'score': k / 100 + 0.02 * split, 'n_test': 30, 'n_train': 70})
    return pd.DataFrame(rows)


def test_a_baseline_with_a_k_list_meets_each_method_at_the_same_k():
    """Test that the method at k = 10 is paired with the baseline at k = 10, not at 5."""
    table = paired_tests.tests(_results([5, 10]), 'score', baseline='Base')

    at_10 = table.set_index('k').loc[10, 'mean_diff']
    expected = np.mean([0.8 + 0.01 * s - (0.10 + 0.02 * s) for s in range(4)])
    assert abs(at_10 - expected) < 1e-12


def test_a_baseline_with_one_setting_meets_every_k():
    """Test that a single-setting baseline, such as all features, is paired with every k."""
    table = paired_tests.tests(_results([2048]), 'score', baseline='Base')

    assert sorted(table['k']) == [5, 10]
    base = [20.48 + 0.02 * s for s in range(4)]
    for k, method in [(5, 0.6), (10, 0.8)]:
        expected = np.mean([method + 0.01 * s - base[s] for s in range(4)])
        assert abs(table.set_index('k').loc[k, 'mean_diff'] - expected) < 1e-12
