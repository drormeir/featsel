"""
Tests for the berk_jones selector.
"""

import numpy as np
import pytest
from scipy.stats import beta, mannwhitneyu

from featsel.selectors import BerkJonesSelector, FeatureSelector
from featsel.selectors.berk_jones import log_beta_cdf


def _direct(X, y, alpha0=0.1):
    """
    The Berk-Jones selection straight from SciPy's Beta CDF, in float64.

    Valid only while every score is above 1e-308.
    """
    p = mannwhitneyu(X[y == 1], X[y == 0], method='asymptotic', axis=0).pvalue
    n = len(p)
    i = np.arange(1, int(alpha0 * n) + 1)
    p_i = np.sort(p)[:len(i)]
    return p <= p_i[np.argmin(beta.cdf(p_i, i, n - i + 1))]


def _rare_weak(n_samples=200, n_features=2000, n_signal=40, shift=0.4, seed=0):
    """Two classes; a few features differ in mean by a small shift, the rest are noise."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_samples // 2)
    X = rng.normal(size=(n_samples, n_features))
    X[y == 1, :n_signal] += shift
    return X, y


class TestLogBetaCdf:
    """Tests for log_beta_cdf, the score in log form."""

    def test_matches_scipy_where_float64_suffices(self):
        """Test that it equals log(betainc) where the CDF is a normal float."""
        a = np.array([1, 5, 50, 500])
        b = 3000 - a + 1
        x = np.array([1e-6, 1e-4, 0.005, 0.15])

        np.testing.assert_allclose(log_beta_cdf(np.log(x), a, b),
                                   np.log(beta.cdf(x, a, b)), rtol=1e-12)

    @pytest.mark.parametrize('log_x, a', [(-800.0, 1), (-60.0, 20), (-3000.0, 5), (-12.0, 300)])
    def test_below_float64(self, log_x, a):
        """Test the series where the CDF is below 1e-300, against 50-digit arithmetic."""
        mpmath = pytest.importorskip('mpmath')
        mpmath.mp.dps = 50
        b = 30000 - a + 1
        exact = mpmath.log(mpmath.betainc(a, b, 0, mpmath.exp(log_x), regularized=True))

        result = log_beta_cdf(log_x, a, b)

        assert result < np.log(1e-300)
        assert result == pytest.approx(float(exact), rel=1e-12)


class TestBerkJonesSelector:
    """Tests for BerkJonesSelector."""

    def test_matches_the_formula(self):
        """Test that it selects what SciPy's Beta CDF selects where float64 suffices."""
        X, y = _rare_weak()

        selector = BerkJonesSelector().fit(X, y)

        np.testing.assert_array_equal(selector.get_support(), _direct(X, y))

    def test_keeps_signal_at_high_precision(self):
        """Test that what it keeps is mostly signal, far above the 2% base rate."""
        X, y = _rare_weak()

        kept = BerkJonesSelector().fit(X, y).get_support(indices=True)

        assert len(kept) >= 1
        assert np.isin(kept, np.arange(40)).mean() > 10 * 40 / 2000

    def test_p_values_below_float64(self):
        """Test that p-values float64 stores as 0 still give a finite, valid choice."""
        X, y = _rare_weak(n_samples=3000, n_signal=50, shift=2.0)
        assert (mannwhitneyu(X[y == 1], X[y == 0], axis=0).pvalue == 0).sum() > 10

        selector = BerkJonesSelector().fit(X, y)

        assert np.isfinite(selector.log_pvalues_).all()
        assert 1 <= selector.get_support().sum() <= 0.1 * X.shape[1]

    def test_keeps_ties_with_the_threshold(self):
        """Test that every feature at the threshold p-value is kept, not some of them."""
        X, y = _rare_weak()
        X = np.hstack([X, X[:, :1]])  # a copy of feature 0, same p-value

        support = BerkJonesSelector().fit(X, y).get_support()

        assert support[0] == support[-1]

    def test_constant_feature_is_never_kept(self):
        """Test that a feature with no variance gets p = 1, log p = 0."""
        X, y = _rare_weak()
        X[:, 0] = 1.0

        selector = BerkJonesSelector().fit(X, y)

        assert selector.log_pvalues_[0] == 0.0
        assert not selector.get_support()[0]

    def test_requires_target(self, small_classification_data):
        """Test that it raises without a target."""
        X, _ = small_classification_data

        with pytest.raises(ValueError, match="requires target"):
            BerkJonesSelector().fit(X, None)

    def test_created_by_name(self):
        """Test that a config can name it by class name, any case."""
        selector = FeatureSelector.create('berkjonesselector', alpha0=0.2)

        assert isinstance(selector, BerkJonesSelector)

    def test_monotone_transform_keeps_the_selection(self):
        """Test that only the order of each feature matters, as for an AUC."""
        X, y = _rare_weak()

        raw = BerkJonesSelector().fit(X, y).get_support()
        transformed = BerkJonesSelector().fit(np.exp(X) ** 3, y).get_support()

        np.testing.assert_array_equal(raw, transformed)

    def test_more_than_two_classes_is_rejected(self):
        """Test that the U test refuses a multiclass target."""
        X, _ = _rare_weak()
        y = np.arange(len(X)) % 3

        with pytest.raises(ValueError, match='two classes'):
            BerkJonesSelector().fit(X, y)
