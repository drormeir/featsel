"""
Tests for the anova_f selector.
"""

import pytest
from sklearn.feature_selection import SelectKBest, f_classif

from featsel.selectors import ANOVAFSelector
from tests.sklearn_reference import assert_not_slower, assert_same_selection


def _with_sklearn():
    """This selector and its scikit-learn equivalent, both unfitted."""
    return ANOVAFSelector(n_features=20), SelectKBest(f_classif, k=20)


class TestANOVAFSelector:
    """Tests for ANOVAFSelector."""

    def test_select_top_features(self, small_classification_data):
        """Test selecting top n features by ANOVA F-score."""
        X, y = small_classification_data

        selector = ANOVAFSelector(n_features=10).fit(X, y)

        assert selector.transform(X).shape == (100, 10)
        assert len(selector.get_support(indices=True)) == 10

    def test_requires_target(self, small_classification_data):
        """Test that ANOVA F requires target variable."""
        X, _ = small_classification_data

        with pytest.raises(ValueError, match="requires target"):
            ANOVAFSelector(n_features=10).fit(X, None)

    def test_classification_vs_regression(self, small_classification_data, small_regression_data):
        """Test that task parameter correctly selects f_classif vs f_regression."""
        X_clf, y_clf = small_classification_data
        X_reg, y_reg = small_regression_data

        clf = ANOVAFSelector(n_features=10, task='classification').fit(X_clf, y_clf)
        reg = ANOVAFSelector(n_features=10, task='regression').fit(X_reg, y_reg)

        assert clf.get_support().sum() == 10
        assert reg.get_support().sum() == 10

    def test_feature_importances(self, small_classification_data):
        """Test that feature importances are computed."""
        X, y = small_classification_data

        selector = ANOVAFSelector(n_features=10).fit(X, y)

        assert len(selector.feature_importances_) == X.shape[1]

    def test_matches_sklearn(self, sklearn_reference_data):
        """Test that it selects the same features as SelectKBest(f_classif)."""
        assert_same_selection(*_with_sklearn(), *sklearn_reference_data)

    def test_not_slower_than_sklearn(self, sklearn_reference_data):
        """Test that fit is at most 5% slower than SelectKBest(f_classif)."""
        assert_not_slower(*_with_sklearn(), *sklearn_reference_data)
