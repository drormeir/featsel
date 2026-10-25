"""
Tests for the higher_criticism selector.
"""

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from featsel.selectors import FeatureSelector, HigherCriticismSelector


def _direct(X, y, variant, alpha0=0.1):
    """
    The HC selection straight from the formulas, in plain float64.

    Valid only while every p-value is above 1e-308.
    """
    p = mannwhitneyu(X[y == 1], X[y == 0], method='asymptotic', axis=0).pvalue
    n = len(p)
    order = np.argsort(p, kind='stable')
    i = np.arange(1, int(alpha0 * n) + 1)
    p_i = p[order[:len(i)]]
    if variant == '2004':
        hc = np.sqrt(n) * (i / n - p_i) / np.sqrt(p_i * (1 - p_i))
    else:
        mu = i / (n + 1)
        hc = np.sqrt(n) * (mu - p_i) / np.sqrt(mu * (1 - mu))
    return p <= p_i[np.argmax(hc)]


def _rare_weak(n_samples=200, n_features=2000, n_signal=40, shift=0.4, seed=0):
    """Two classes; a few features differ in mean by a small shift, the rest are noise."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_samples // 2)
    X = rng.normal(size=(n_samples, n_features))
    X[y == 1, :n_signal] += shift
    return X, y


class TestHigherCriticismSelector:
    """Tests for HigherCriticismSelector."""

    @pytest.mark.parametrize('variant', ['2004', '2008'])
    def test_matches_the_formula(self, variant):
        """Test that it selects what the formula selects where float64 suffices."""
        X, y = _rare_weak()

        selector = HigherCriticismSelector(variant=variant).fit(X, y)

        np.testing.assert_array_equal(selector.get_support(), _direct(X, y, variant))

    def test_finds_rare_weak_signal(self):
        """
        Test that most signal is kept, at a precision far above chance.

        HC trades false features for missed ones (Donoho and Jin 2008), so
        precision is checked against the 2% base rate, not against 50%.
        """
        X, y = _rare_weak()

        kept = HigherCriticismSelector().fit(X, y).get_support(indices=True)
        signal = np.isin(kept, np.arange(40))

        assert signal.sum() / 40 > 0.5
        assert signal.mean() > 10 * 40 / 2000

    @pytest.mark.parametrize('variant', ['2004', '2008'])
    def test_p_values_below_float64(self, variant):
        """Test that p-values float64 stores as 0 still give a finite, valid choice."""
        X, y = _rare_weak(n_samples=3000, n_signal=50, shift=2.0)
        assert (mannwhitneyu(X[y == 1], X[y == 0], axis=0).pvalue == 0).sum() > 10

        selector = HigherCriticismSelector(variant=variant).fit(X, y)

        assert np.isfinite(selector.log_pvalues_).all()
        assert 1 <= selector.get_support().sum() <= 0.1 * X.shape[1]

    def test_keeps_ties_with_the_threshold(self):
        """Test that every feature at the threshold p-value is kept, not some of them."""
        X, y = _rare_weak()
        X = np.hstack([X, X[:, :1]])  # a copy of feature 0, same p-value

        support = HigherCriticismSelector().fit(X, y).get_support()

        assert support[0] == support[-1]

    def test_constant_feature_is_never_kept(self):
        """Test that a feature with no variance gets p = 1, log p = 0."""
        X, y = _rare_weak()
        X[:, 0] = 1.0

        selector = HigherCriticismSelector().fit(X, y)

        assert selector.log_pvalues_[0] == 0.0
        assert not selector.get_support()[0]

    def test_unknown_variant_is_rejected(self, small_classification_data):
        """Test that a misspelled variant fails loudly."""
        X, y = small_classification_data

        with pytest.raises(ValueError, match="variant must be"):
            HigherCriticismSelector(variant='2010').fit(X, y)

    def test_requires_target(self, small_classification_data):
        """Test that HC raises without a target."""
        X, _ = small_classification_data

        with pytest.raises(ValueError, match="requires target"):
            HigherCriticismSelector().fit(X, None)

    def test_created_by_name(self):
        """Test that a config can name it by class name, any case."""
        selector = FeatureSelector.create('higherCriticismSelector', variant='2004')

        assert isinstance(selector, HigherCriticismSelector)

    def test_monotone_transform_keeps_the_selection(self):
        """Test that only the order of each feature matters, as for an AUC."""
        X, y = _rare_weak()

        raw = HigherCriticismSelector().fit(X, y).get_support()
        transformed = HigherCriticismSelector().fit(np.exp(X) ** 3, y).get_support()

        np.testing.assert_array_equal(raw, transformed)

    def test_more_than_two_classes_is_rejected(self):
        """Test that the U test refuses a multiclass target."""
        X, _ = _rare_weak()
        y = np.arange(len(X)) % 3

        with pytest.raises(ValueError, match='two classes'):
            HigherCriticismSelector().fit(X, y)
