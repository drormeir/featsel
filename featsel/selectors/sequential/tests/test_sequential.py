"""
Tests for the sequential selector.
"""

import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

from featsel.selectors import SequentialSelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return (
        SequentialSelector(n_features=3, cv=3),
        SequentialFeatureSelector(LogisticRegression(C=1.0, max_iter=1000),
                                  n_features_to_select=3, cv=3),
    )


@pytest.fixture(scope='module')
def narrow_data():
    """
    A narrow dataset for the scikit-learn comparison.

    Sequential selection fits one model per candidate feature per step, so the
    shared 200-feature dataset would take minutes per fit.
    """
    return make_classification(n_samples=60, n_features=8, n_informative=3,
                               n_redundant=0, random_state=0)


class TestSequentialSelector:
    """Tests for SequentialSelector."""

    def test_forward_keeps_n_features(self, narrow_data):
        """Test that forward selection stops at n_features."""
        X, y = narrow_data

        selector = SequentialSelector(n_features=3, cv=3).fit(X, y)

        assert selector.transform(X).shape == (60, 3)

    def test_backward_keeps_n_features(self, narrow_data):
        """Test that backward selection stops at n_features."""
        X, y = narrow_data

        selector = SequentialSelector(n_features=3, direction='backward', cv=3).fit(X, y)

        assert selector.get_support().sum() == 3

    def test_requires_n_features(self, narrow_data):
        """Test that sequential selection needs a target count."""
        X, y = narrow_data

        with pytest.raises(ValueError, match="requires n_features"):
            SequentialSelector().fit(X, y)

    def test_matches_sklearn(self, narrow_data):
        """Test that it selects the same features as SequentialFeatureSelector."""
        assert_same_selection(*_with_sklearn(), *narrow_data)

    def test_not_slower_than_sklearn(self, narrow_data):
        """Test that fit is at most 5% slower than SequentialFeatureSelector."""
        assert_not_slower(*_with_sklearn(), *narrow_data)
