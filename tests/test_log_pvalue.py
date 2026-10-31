"""
Tests for the U-test and F-test log p-values.
"""

import mpmath
import numpy as np
import pytest
from scipy.stats import f as f_dist
from scipy.stats import mannwhitneyu
from scipy.stats import t as t_dist

from featsel.selectors.log_pvalue import log_f_sf, log_mannwhitneyu

# Two classes and 3,000 patients, as in a SCAN-B subtype-vs-rest task.
DFN, DFD = 1, 2998


def _exact(f, dfn, dfd):
    """log p by mpmath at 50 digits: the regularized incomplete beta, no underflow."""
    mpmath.mp.dps = 50
    x = mpmath.mpf(dfd) / (dfd + dfn * mpmath.mpf(f))
    p = mpmath.betainc(mpmath.mpf(dfd) / 2, mpmath.mpf(dfn) / 2, 0, x, regularized=True)
    return float(mpmath.log(p))


class TestLogFSf:
    """Tests for log_f_sf."""

    @pytest.mark.parametrize('dfn, dfd', [(1, 2998), (4, 2970), (1, 778)])
    def test_matches_scipy_where_scipy_is_exact(self, dfn, dfd):
        """Test that it equals log of scipy's p-value while that is above 1e-300."""
        f = np.array([0.01, 0.5, 1.0, 3.0, 10.0, 50.0, 200.0])
        expected = f_dist.logsf(f, dfn, dfd)

        np.testing.assert_allclose(log_f_sf(f, dfn, dfd), expected, rtol=1e-10)

    @pytest.mark.parametrize('f', [5e3, 1e4, 1e5, 1e7])
    def test_matches_mpmath_below_float64(self, f):
        """Test that p-values scipy reports as 0 match the exact value."""
        assert f_dist.sf(f, DFN, DFD) == 0.0

        assert log_f_sf([f], DFN, DFD)[0] == pytest.approx(_exact(f, DFN, DFD), rel=1e-10)

    def test_matches_mpmath_with_more_classes(self):
        """Test the series on a five-class F, numerator df 4."""
        assert log_f_sf([2000.0], 4, 2970)[0] == pytest.approx(_exact(2000.0, 4, 2970),
                                                               rel=1e-10)

    def test_equals_two_sided_t_test(self):
        """Test that with two classes it is the two-sided t-test p-value, F = t^2."""
        t = np.array([0.5, 2.0, 5.0, 20.0])

        expected = np.log(2) + t_dist.logsf(t, DFD)

        np.testing.assert_allclose(log_f_sf(t ** 2, DFN, DFD), expected, rtol=1e-10)

    def test_many_genes_at_once_equal_one_at_a_time(self):
        """Test that a mix of weak and strong genes converges, element by element."""
        f = np.random.default_rng(0).chisquare(1, 3000)
        f[:100] *= 2000.0

        together = log_f_sf(f, DFN, DFD)
        alone = [log_f_sf([value], DFN, DFD)[0] for value in f[::30]]

        np.testing.assert_array_equal(together[::30], alone)

    def test_decreases_with_f(self):
        """Test that a larger F always gives a smaller p-value."""
        f = np.logspace(-3, 8, 200)

        assert np.all(np.diff(log_f_sf(f, DFN, DFD)) < 0)

    def test_edge_values(self):
        """Test F = 0, F = inf and NaN, as f_classif can return all three."""
        result = log_f_sf([0.0, np.inf, np.nan], DFN, DFD)

        assert result[0] == 0.0
        assert result[1] == -np.inf
        assert np.isnan(result[2])


class TestLogMannWhitneyU:
    """Tests for log_mannwhitneyu."""

    def test_matches_scipy_with_ties(self):
        """Test p-values and AUC against scipy, on features with many tied values."""
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        X = np.round(rng.normal(size=(400, 60)), 1)
        X[:, :20] = np.maximum(X[:, :20], 0)  # half the values tied at 0
        X[y == 1, 20:30] += 0.6

        log_p, auc = log_mannwhitneyu(X, y)
        expected = mannwhitneyu(X[y == 1], X[y == 0], method='asymptotic', axis=0)

        np.testing.assert_allclose(log_p, np.log(expected.pvalue), rtol=1e-12)
        np.testing.assert_allclose(auc, expected.statistic / ((y == 1).sum() * (y == 0).sum()))

    def test_finite_where_scipy_gives_zero(self):
        """Test that a p-value below 1e-308 keeps its log."""
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1], 1500)
        X = rng.normal(size=(3000, 3))
        X[y == 1] += 3
        assert (mannwhitneyu(X[y == 1], X[y == 0], axis=0).pvalue == 0).all()

        log_p, _ = log_mannwhitneyu(X, y)

        assert np.isfinite(log_p).all()
        assert (log_p < np.log(1e-308)).all()

    def test_constant_feature_has_p_one(self):
        """Test that a feature with one value gets p = 1, log p = 0."""
        y = np.repeat([0, 1], 10)
        X = np.ones((20, 1))

        log_p, _ = log_mannwhitneyu(X, y)

        assert (log_p == 0.0).all()
