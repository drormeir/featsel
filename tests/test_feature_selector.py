"""
Tests for FeatureSelector and filter methods.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score

from featsel import FeatureSelector


class TestVarianceThreshold:
    """Tests for VarianceThreshold selector."""

    def test_remove_constant_features(self, data_with_constant_features):
        """Test that constant features are removed."""
        X, y = data_with_constant_features

        selector = FeatureSelector(method='variance_threshold', threshold=0.0)
        selector.fit(X)

        X_selected = selector.transform(X)

        # Should remove 3 constant features
        assert X_selected.shape[1] == 12  # 15 - 3 = 12
        assert not any('constant' in name for name in X_selected.columns)

    def test_remove_low_variance(self, data_with_constant_features):
        """Test that low-variance features are removed."""
        X, y = data_with_constant_features

        selector = FeatureSelector(method='variance_threshold', threshold=0.1)
        selector.fit(X)

        X_selected = selector.transform(X)

        # Should remove 3 constant + 2 low-variance features
        assert X_selected.shape[1] == 10
        assert not any('constant' in name for name in X_selected.columns)
        assert not any('low_var' in name for name in X_selected.columns)

    def test_dataframe_preservation(self, small_classification_data):
        """Test that DataFrame input returns DataFrame output."""
        X, y = small_classification_data

        selector = FeatureSelector(method='variance_threshold')
        X_selected = selector.fit_transform(X)

        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape[0] == X.shape[0]


class TestANOVAFSelector:
    """Tests for ANOVAFSelector."""

    def test_select_top_features(self, small_classification_data):
        """Test selecting top n features by ANOVA F-score."""
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)
        selector.fit(X, y)

        X_selected = selector.transform(X)

        assert X_selected.shape == (100, 10)
        assert len(selector.selected_features_) == 10

    def test_requires_target(self, small_classification_data):
        """Test that ANOVA F requires target variable."""
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)

        with pytest.raises(ValueError, match="requires target"):
            selector.fit(X)  # Missing y

    def test_classification_vs_regression(self, small_classification_data, small_regression_data):
        """Test that task parameter correctly selects f_classif vs f_regression."""
        X_clf, y_clf = small_classification_data
        X_reg, y_reg = small_regression_data

        # Classification task
        selector_clf = FeatureSelector(method='anova_f', n_features=10, task='classification')
        selector_clf.fit(X_clf, y_clf)
        assert selector_clf.n_features_out_ == 10

        # Regression task
        selector_reg = FeatureSelector(method='anova_f', n_features=10, task='regression')
        selector_reg.fit(X_reg, y_reg)
        assert selector_reg.n_features_out_ == 10

    def test_feature_importances(self, small_classification_data):
        """Test that feature importances are computed."""
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)
        selector.fit(X, y)

        assert selector.feature_importances_ is not None
        assert len(selector.feature_importances_) == X.shape[1]


class TestMutualInfoSelector:
    """Tests for MutualInfoSelector."""

    def test_select_features(self, small_classification_data):
        """Test mutual information feature selection."""
        X, y = small_classification_data

        selector = FeatureSelector(
            method='mutual_info',
            n_features=10,
            random_state=42
        )
        selector.fit(X, y)

        X_selected = selector.transform(X)

        assert X_selected.shape == (100, 10)
        assert len(selector.selected_features_) == 10

    def test_requires_n_features(self, small_classification_data):
        """Test that mutual info requires n_features parameter."""
        X, y = small_classification_data

        selector = FeatureSelector(method='mutual_info')  # Missing n_features

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            selector.fit(X, y)

    def test_reproducibility(self, small_classification_data):
        """Test that random_state ensures reproducibility."""
        X, y = small_classification_data

        selector1 = FeatureSelector(method='mutual_info', n_features=10, random_state=42)
        selector1.fit(X, y)

        selector2 = FeatureSelector(method='mutual_info', n_features=10, random_state=42)
        selector2.fit(X, y)

        assert selector1.selected_features_ == selector2.selected_features_


class TestCorrelationSelector:
    """Tests for CorrelationSelector."""

    def test_select_correlated_features(self, small_regression_data):
        """Test correlation-based selection."""
        X, y = small_regression_data

        selector = FeatureSelector(
            method='correlation',
            n_features=10,
            target_threshold=0.0
        )
        selector.fit(X, y)

        X_selected = selector.transform(X)

        assert X_selected.shape[1] == 10
        assert len(selector.selected_features_) == 10

    def test_remove_redundant_features(self):
        """Test that highly correlated features are removed."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        # Create redundant feature (copy of feature 0)
        X_redundant = np.column_stack([X, X[:, 0] + np.random.randn(100) * 0.01])

        X_df = pd.DataFrame(X_redundant, columns=[f'f{i}' for i in range(6)])
        y = X[:, 0] + X[:, 1]

        selector = FeatureSelector(
            method='correlation',
            inter_feature_threshold=0.9
        )
        selector.fit(X_df, y)

        # Should remove one of the redundant features
        assert selector.n_features_out_ < 6


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
        """Test that FeatureSelector can handle methods not yet implemented."""
        X, y = high_dim_data

        # This will fail because we haven't implemented lasso yet
        # but it tests error handling
        with pytest.raises(ValueError, match="Unknown method"):
            selector = FeatureSelector(method='lasso', n_features=50)
            selector.fit(X, y)


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
        X, y = small_classification_data

        selector = FeatureSelector(method='anova_f', n_features=10)

        with pytest.raises(Exception):  # sklearn raises NotFittedError
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
