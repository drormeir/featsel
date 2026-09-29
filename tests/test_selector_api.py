"""
Tests for the interface every selector shares: scikit-learn compatibility.

Each selector's own behaviour is tested next to it, in
featsel/selectors/<name>/tests.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from featsel.selectors import (
    ANOVAFSelector,
    CorrelationSelector,
    LassoSelector,
    MutualInfoSelector,
    RandomSelector,
    TreeImportanceSelector,
    VarianceThreshold,
)

# One configured instance of every selector, each keeping 10 features or fewer.
SELECTORS = [
    ANOVAFSelector(n_features=10),
    CorrelationSelector(n_features=10, target_threshold=0.0),
    LassoSelector(n_features=10, random_state=0),
    MutualInfoSelector(n_features=10, random_state=0),
    RandomSelector(n_features=10, random_state=0),
    TreeImportanceSelector(n_features=10, n_estimators=20, random_state=0),
    VarianceThreshold(threshold=0.0),
]
IDS = [type(s).__name__ for s in SELECTORS]


@pytest.mark.parametrize('selector', SELECTORS, ids=IDS)
def test_get_support_mask_and_indices(selector, small_classification_data):
    """Test that get_support gives a boolean mask or the matching indices."""
    X, y = small_classification_data
    fitted = clone(selector).fit(X, y)

    mask = fitted.get_support()
    indices = fitted.get_support(indices=True)

    assert mask.dtype == bool
    assert len(mask) == X.shape[1]
    assert np.array_equal(np.flatnonzero(mask), indices)
    assert fitted.transform(X).shape == (X.shape[0], mask.sum())


@pytest.mark.parametrize('selector', SELECTORS, ids=IDS)
def test_clone_keeps_parameters(selector):
    """Test that clone() rebuilds an identical, unfitted selector."""
    copy = clone(selector)

    assert copy.get_params() == selector.get_params()
    assert not hasattr(copy, 'support_')


@pytest.mark.parametrize('selector', SELECTORS, ids=IDS)
def test_cross_val_score_in_a_pipeline(selector, small_classification_data):
    """Test that selection refits inside each fold of scikit-learn's cross-validation."""
    X, y = small_classification_data
    pipe = Pipeline([('select', selector), ('clf', LogisticRegression(max_iter=1000))])

    scores = cross_val_score(pipe, X, y, cv=3)

    assert len(scores) == 3
    assert all(score > 0.5 for score in scores)


@pytest.mark.parametrize('selector', SELECTORS, ids=IDS)
def test_skipping_input_check_keeps_the_selection(selector, small_classification_data):
    """Test that check_input=False selects the same features from clean arrays."""
    X, y = small_classification_data
    checked = clone(selector).fit(X.values, y.values)
    unchecked = clone(selector).set_params(check_input=False).fit(X.values, y.values)

    assert np.array_equal(checked.get_support(), unchecked.get_support())


@pytest.mark.parametrize('selector', SELECTORS, ids=IDS)
def test_transform_before_fit_raises(selector, small_classification_data):
    """Test that an unfitted selector refuses to transform."""
    X, _ = small_classification_data

    with pytest.raises(NotFittedError):
        clone(selector).transform(X)


def test_grid_search_tunes_n_features(small_classification_data):
    """Test that n_features is a searchable parameter."""
    X, y = small_classification_data
    pipe = Pipeline([('select', ANOVAFSelector()), ('clf', LogisticRegression(max_iter=1000))])

    search = GridSearchCV(pipe, {'select__n_features': [2, 5, 10]}, cv=3).fit(X, y)

    assert search.best_params_['select__n_features'] in (2, 5, 10)


def test_set_params_changes_the_selection(small_classification_data):
    """Test that set_params() reaches the selector's behaviour."""
    X, y = small_classification_data
    selector = ANOVAFSelector(n_features=10).set_params(n_features=3)

    assert selector.fit(X, y).get_support().sum() == 3


def test_feature_names_follow_dataframe_columns(small_classification_data):
    """Test that selected column names come back for DataFrame input."""
    X, y = small_classification_data
    selector = ANOVAFSelector(n_features=5).fit(X, y)

    names = selector.get_feature_names_out()

    assert list(names) == list(X.columns[selector.get_support()])


def test_pandas_output_keeps_the_dataframe(small_classification_data):
    """Test that set_output(transform='pandas') returns the selected columns."""
    X, y = small_classification_data
    selector = ANOVAFSelector(n_features=5).set_output(transform='pandas').fit(X, y)

    X_selected = selector.transform(X)

    assert isinstance(X_selected, pd.DataFrame)
    assert list(X_selected.columns) == list(selector.get_feature_names_out())


def test_numpy_input(small_classification_data):
    """Test that plain arrays work and give arrays back."""
    X, y = small_classification_data

    X_selected = ANOVAFSelector(n_features=10).fit_transform(X.values, y.values)

    assert isinstance(X_selected, np.ndarray)
    assert X_selected.shape == (100, 10)


def test_misspelled_parameter_is_rejected():
    """Test that a typo in a parameter name fails instead of being ignored."""
    with pytest.raises(TypeError):
        ANOVAFSelector(n_featurs=10)


class TestHighDimensionalData:
    """Tests for high-dimensional data (n_features >> n_samples)."""

    def test_anova_on_high_dim(self, high_dim_data):
        """Test univariate selection when features far outnumber samples."""
        X, y = high_dim_data  # 50 samples, 500 features

        assert ANOVAFSelector(n_features=50).fit_transform(X, y).shape == (50, 50)

    def test_lasso_on_high_dim(self, high_dim_data):
        """Test Lasso selection when features far outnumber samples."""
        X, y = high_dim_data

        X_selected = LassoSelector(n_features=50, random_state=42).fit_transform(X, y)

        assert X_selected.shape == (X.shape[0], 50)

    def test_lasso_multiclass_on_high_dim(self, high_dim_data):
        """Test that multiclass targets work despite liblinear being binary-only."""
        X, _ = high_dim_data
        y_multi = np.tile([0, 1, 2, 3, 4], X.shape[0] // 5 + 1)[:X.shape[0]]

        selector = LassoSelector(n_features=20, random_state=42).fit(X, y_multi)

        assert selector.coef_.shape[0] == 5
        assert selector.get_support().sum() == 20


def test_embedded_methods_agree_on_informative_data(small_classification_data):
    """Test that Lasso and tree importance overlap more than chance would."""
    X, y = small_classification_data

    lasso = LassoSelector(n_features=10, random_state=42).fit(X, y)
    trees = TreeImportanceSelector(n_features=10, n_estimators=50, random_state=42).fit(X, y)

    overlap = set(lasso.get_support(indices=True)) & set(trees.get_support(indices=True))
    assert len(overlap) >= 2


def test_selection_is_independent_of_the_target_switch(scanb_loader):
    """Test that a fitted unsupervised selector transforms data under any target."""
    loader = scanb_loader
    selector = VarianceThreshold(threshold=0.1).fit(loader.X)

    loader.set_target('PAM50')
    first = selector.transform(loader.X)
    loader.set_target('ER')
    second = selector.transform(loader.X)

    assert first.shape == second.shape
