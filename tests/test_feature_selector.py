"""
Tests for the FeatureSelector wrapper.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from featsel import FeatureSelector


class TestEmbeddedSelectors:
    """Tests comparing the embedded methods with each other."""

    def test_embedded_beats_random_on_informative_data(self, small_classification_data):
        """Test that embedded methods find informative features more often than chance."""
        X, y = small_classification_data

        lasso = FeatureSelector(method='lasso', n_features=10, random_state=42).fit(X, y)
        trees = FeatureSelector(method='tree_importance', n_features=10,
                                n_estimators=50, random_state=42).fit(X, y)

        overlap = set(lasso.selected_features_) & set(trees.selected_features_)
        assert len(overlap) >= 2


class TestFeatureSelectorAPI:
    """Tests for FeatureSelector main API."""

    def test_sklearn_pipeline_integration(self, small_classification_data):
        """Test FeatureSelector in sklearn Pipeline."""
        X, y = small_classification_data

        pipe = Pipeline([
            ('select', FeatureSelector(method='anova_f', n_features=10)),
            ('clf', LogisticRegression(random_state=42))
        ])

        pipe.fit(X, y)
        score = pipe.score(X, y)

        assert score > 0.5  # Sanity check
        assert pipe.named_steps['select'].n_features_out_ == 10

    def test_get_support_mask(self, small_classification_data):
        """Test get_support() returns boolean mask."""
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)
        selector.fit(X, y)

        support_mask = selector.get_support(indices=False)

        assert isinstance(support_mask, np.ndarray)
        assert support_mask.dtype == bool
        assert len(support_mask) == X.shape[1]
        assert np.sum(support_mask) == 10

    def test_get_support_indices(self, small_classification_data):
        """Test get_support() returns integer indices."""
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)
        selector.fit(X, y)

        support_indices = selector.get_support(indices=True)

        assert isinstance(support_indices, np.ndarray)
        assert support_indices.dtype in [np.int32, np.int64]
        assert len(support_indices) == 10

    def test_get_feature_names_out(self, small_classification_data):
        """Test get_feature_names_out() returns selected feature names."""
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)
        selector.fit(X, y)

        feature_names = selector.get_feature_names_out()

        assert len(feature_names) == 10
        assert all(isinstance(name, str) for name in feature_names)
        assert all('feature_' in name for name in feature_names)

    def test_fit_transform(self, small_classification_data):
        """Test fit_transform() method."""
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (100, 10)
        assert isinstance(X_selected, pd.DataFrame)

    def test_get_report(self, small_classification_data):
        """Test get_report() generates feature report."""
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)
        selector.fit(X, y)

        report = selector.get_report()

        assert isinstance(report, pd.DataFrame)
        assert len(report) == X.shape[1]
        assert 'feature_name' in report.columns
        assert 'selected' in report.columns
        assert 'importance_score' in report.columns
        assert 'rank' in report.columns
        assert report['selected'].sum() == 10

    def test_numpy_array_input(self, small_classification_data):
        """Test that numpy arrays work as input."""
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)
        X_selected = selector.fit_transform(X.values, y.values)

        assert isinstance(X_selected, np.ndarray)
        assert X_selected.shape == (100, 10)


class TestHighDimensionalData:
    """Tests for high-dimensional data (n_features >> n_samples)."""

    def test_select_from_high_dim(self, high_dim_data):
        """Test feature selection on high-dimensional data."""
        X, y = high_dim_data  # 50 samples, 500 features

        selector = FeatureSelector(method='anova_f', n_features=50)
        selector.fit(X, y)

        X_selected = selector.transform(X)

        assert X_selected.shape == (50, 50)
        assert len(selector.selected_features_) == 50

    def test_lasso_on_high_dim(self, high_dim_data):
        """Test Lasso selection when features far outnumber samples."""
        X, y = high_dim_data

        selector = FeatureSelector(method='lasso', n_features=50, random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (X.shape[0], 50)

    def test_lasso_multiclass_on_high_dim(self, high_dim_data):
        """Test that multiclass targets work despite liblinear being binary-only."""
        X, _ = high_dim_data
        y_multi = pd.Series(np.tile([0, 1, 2, 3, 4], X.shape[0] // 5 + 1)[:X.shape[0]],
                            index=X.index)

        selector = FeatureSelector(method='lasso', n_features=20, random_state=42)
        selector.fit(X, y_multi)

        assert selector.selector_.coef_.shape[0] == 5
        assert selector.n_features_out_ == 20


class TestMultipleTargets:
    """Tests for multi-target support (with DataLoader)."""

    def test_feature_selection_preserves_multi_target(self, scanb_loader):
        """Test that feature selection works with DataLoader multi-target support."""
        loader = scanb_loader

        # Fit on PAM50
        loader.set_target('PAM50')
        selector = FeatureSelector(method='variance_threshold', threshold=0.1)
        selector.fit(loader.X)

        # Transform should work regardless of current target
        X_selected = selector.transform(loader.X)

        # Switch target
        loader.set_target('ER')

        # Transform should still work
        X_selected2 = selector.transform(loader.X)

        assert X_selected.shape == X_selected2.shape
        assert (X_selected.columns == X_selected2.columns).all()


class TestErrorHandling:
    """Tests for error handling."""

    def test_transform_before_fit(self, small_classification_data):
        """Test that transform before fit raises error."""
        X, _y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)

        with pytest.raises(NotFittedError):
            selector.transform(X)

    def test_invalid_method(self, small_classification_data):
        """Test that invalid method name raises error."""
        X, y = small_classification_data

        selector = FeatureSelector(method='invalid_method')

        with pytest.raises(ValueError, match="Unknown method"):
            selector.fit(X, y)

    def test_invalid_input_type(self):
        """Test that invalid input type raises error."""
        selector = FeatureSelector(method='anova_f', n_features=10)

        with pytest.raises(TypeError):
            selector.fit([1, 2, 3], [0, 1, 0])  # Lists not supported


class TestCrossValidation:
    """Tests for cross-validation with FeatureSelector."""

    def test_cross_val_score(self, small_classification_data):
        """Test that FeatureSelector works with cross-validation."""
        X, y = small_classification_data

        pipe = Pipeline([
            ('select', FeatureSelector(method='anova_f', n_features=10)),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])

        scores = cross_val_score(pipe, X, y, cv=3)

        assert len(scores) == 3
        assert all(score > 0.5 for score in scores)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
