"""
Main feature selection API with sklearn pipeline integration.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Union, List, Dict, Optional

from .selectors import (
    VarianceThreshold,
    ANOVAFSelector,
    MutualInfoSelector,
    CorrelationSelector
)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Unified feature selection transformer compatible with sklearn pipelines.

    This class provides a consistent interface for applying various feature selection
    methods within sklearn pipelines. It supports both single and multiple selection
    methods with different combination strategies.

    Parameters
    ----------
    method : str or list of str, default='variance_threshold'
        Selection method(s) to use. Available methods:
        - 'variance_threshold': Remove low-variance features
        - 'anova_f': ANOVA F-test (classification) or F-test (regression)
        - 'mutual_info': Mutual information with target
        - 'correlation': Correlation-based selection

    n_features : int, optional
        Number of features to select. If None, method-specific default is used.

    threshold : float, optional
        Selection threshold for applicable methods (e.g., variance_threshold).

    task : str, default='classification'
        Type of task: 'classification' or 'regression'.
        Determines which statistical tests to use.

    strategy : str, default='single'
        Strategy when using multiple methods:
        - 'single': Use only the first method (default)
        - 'union': Features selected by ANY method
        - 'intersection': Features selected by ALL methods
        - 'voting': Features selected by majority of methods

    keep_names : bool, default=True
        Store feature names (required for DataFrames, adds overhead for arrays).

    **method_params : dict
        Method-specific parameters. Examples:
        - alpha=0.05 for anova_f
        - n_neighbors=3 for mutual_info
        - target_threshold=0.1 for correlation

    Attributes
    ----------
    selected_features_ : list
        Names or indices of selected features (set after fit()).

    feature_importances_ : np.ndarray or dict
        Feature importance scores. For single method, returns array.
        For multiple methods, returns dict mapping method names to arrays.

    n_features_in_ : int
        Number of features seen during fit.

    n_features_out_ : int
        Number of features selected.

    feature_names_in_ : list or None
        Feature names if input was DataFrame, None otherwise.

    selector_ : BaseSelector or list of BaseSelector
        Fitted selector instance(s).

    Examples
    --------
    Programmatic usage with sklearn pipeline:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> from featsel import FeatureSelector, DataLoader
    >>>
    >>> loader = DataLoader('configs/scanb_small.yaml')
    >>> pipe = Pipeline([
    ...     ('select', FeatureSelector(method='mutual_info', n_features=100)),
    ...     ('clf', LogisticRegression())
    ... ])
    >>> pipe.fit(loader.X, loader.y)
    >>> score = pipe.score(X_test, y_test)

    Multiple methods with voting:

    >>> selector = FeatureSelector(
    ...     method=['mutual_info', 'anova_f', 'correlation'],
    ...     strategy='voting',
    ...     n_features=100
    ... )
    >>> selector.fit(X_train, y_train)
    >>> X_selected = selector.transform(X_test)

    Works with DataLoader:

    >>> from featsel import DataLoader, FeatureSelector
    >>> loader = DataLoader('configs/scanb_small.yaml')
    >>> selector = FeatureSelector(method='correlation', n_features=50)
    >>> selector.fit(loader.X, loader.y)
    >>> X_selected = selector.transform(loader.X)
    >>> print(f"Selected features: {selector.selected_features_}")
    """

    # Mapping from method names to selector classes
    _METHOD_MAP = {
        'variance_threshold': VarianceThreshold,
        'anova_f': ANOVAFSelector,
        'mutual_info': MutualInfoSelector,
        'correlation': CorrelationSelector,
    }

    def __init__(
        self,
        method: Union[str, List[str]] = 'variance_threshold',
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        task: str = 'classification',
        strategy: str = 'single',
        keep_names: bool = True,
        **method_params
    ):
        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.task = task
        self.strategy = strategy
        self.keep_names = keep_names
        self.method_params = method_params

    def fit(self, X, y=None):
        """
        Learn feature selection from training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,), optional
            Target values. Required for supervised methods.

        Returns
        -------
        self : FeatureSelector
            Fitted selector.
        """
        # Store feature info
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = len(self.feature_names_in_)
        elif isinstance(X, np.ndarray):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = None if not self.keep_names else \
                [f'feature_{i}' for i in range(self.n_features_in_)]
        else:
            raise TypeError(f"X must be pandas DataFrame or numpy array, got {type(X)}")

        # Handle single vs multiple methods
        methods = [self.method] if isinstance(self.method, str) else self.method

        # Fit selector(s)
        if len(methods) == 1 or self.strategy == 'single':
            self.selector_ = self._fit_single_method(methods[0], X, y)
        else:
            self.selector_ = [self._fit_single_method(m, X, y) for m in methods]
            self._combine_selections(X)

        return self

    def _fit_single_method(self, method_name, X, y):
        """Fit a single selection method."""
        if method_name not in self._METHOD_MAP:
            available = ', '.join(self._METHOD_MAP.keys())
            raise ValueError(
                f"Unknown method '{method_name}'. "
                f"Available methods: {available}"
            )

        SelectorClass = self._METHOD_MAP[method_name]

        # Build kwargs for this method
        kwargs = self._build_method_kwargs(method_name)

        # Create and fit selector
        selector = SelectorClass(**kwargs)
        selector.fit(X, y)

        return selector

    def _build_method_kwargs(self, method_name):
        """Build kwargs dictionary for a specific method."""
        kwargs = {}

        # Common parameters
        if self.n_features is not None:
            kwargs['n_features'] = self.n_features

        # Method-specific parameters
        if method_name == 'variance_threshold':
            if self.threshold is not None:
                kwargs['threshold'] = self.threshold
            if 'threshold' in self.method_params:
                kwargs['threshold'] = self.method_params['threshold']

        elif method_name == 'anova_f':
            kwargs['task'] = self.task
            if 'alpha' in self.method_params:
                kwargs['alpha'] = self.method_params['alpha']

        elif method_name == 'mutual_info':
            kwargs['task'] = self.task
            if 'n_neighbors' in self.method_params:
                kwargs['n_neighbors'] = self.method_params['n_neighbors']
            if 'random_state' in self.method_params:
                kwargs['random_state'] = self.method_params['random_state']

        elif method_name == 'correlation':
            if 'target_threshold' in self.method_params:
                kwargs['target_threshold'] = self.method_params['target_threshold']
            if 'inter_feature_threshold' in self.method_params:
                kwargs['inter_feature_threshold'] = self.method_params['inter_feature_threshold']
            if 'method' in self.method_params:
                kwargs['method'] = self.method_params['method']

        return kwargs

    def _combine_selections(self, X):
        """Combine feature selections from multiple methods."""
        # Get support masks from all selectors
        support_masks = [selector.get_support() for selector in self.selector_]

        # Combine based on strategy
        if self.strategy == 'union':
            # Features selected by ANY method
            combined_support = np.any(support_masks, axis=0)
        elif self.strategy == 'intersection':
            # Features selected by ALL methods
            combined_support = np.all(support_masks, axis=0)
        elif self.strategy == 'voting':
            # Features selected by majority of methods
            vote_counts = np.sum(support_masks, axis=0)
            threshold = len(self.selector_) / 2
            combined_support = vote_counts > threshold
        else:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                "Available: 'single', 'union', 'intersection', 'voting'"
            )

        # Apply n_features limit if specified
        if self.n_features is not None:
            n_selected = np.sum(combined_support)
            if n_selected > self.n_features:
                # Need to reduce selection - use average importance scores
                avg_importances = self._compute_average_importances()
                selected_indices = np.where(combined_support)[0]
                scores = avg_importances[selected_indices]
                top_indices = selected_indices[np.argsort(scores)[-self.n_features:]]
                combined_support = np.zeros(self.n_features_in_, dtype=bool)
                combined_support[top_indices] = True

        self._combined_support = combined_support

    def _compute_average_importances(self):
        """Compute average importance scores across methods."""
        importances_list = []
        for selector in self.selector_:
            imp = selector.get_feature_importances()
            if imp is not None:
                # Normalize to [0, 1]
                imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-10)
                importances_list.append(imp_norm)

        if not importances_list:
            # No methods provide importance scores, use uniform
            return np.ones(self.n_features_in_)

        return np.mean(importances_list, axis=0)

    def transform(self, X):
        """
        Reduce X to selected features.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : pd.DataFrame or np.ndarray of shape (n_samples, n_features_out)
            Data with only selected features. Type matches input type.
        """
        check_is_fitted(self, 'selector_')

        support = self.get_support()

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, support]
        else:
            return X[:, support]

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step (sklearn standard).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training data.
        y : pd.Series or np.ndarray, optional
            Target values.

        Returns
        -------
        X_selected : pd.DataFrame or np.ndarray
            Transformed data with selected features.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        """
        Get boolean mask or integer indices of selected features (sklearn API).

        Parameters
        ----------
        indices : bool, default=False
            If True, return integer indices of selected features.
            If False, return boolean mask.

        Returns
        -------
        support : np.ndarray
            Boolean mask or integer indices of selected features.
        """
        check_is_fitted(self, 'selector_')

        if isinstance(self.selector_, list):
            # Multiple methods - use combined support
            support = self._combined_support
        else:
            # Single method
            support = self.selector_.get_support(indices=False)

        return np.where(support)[0] if indices else support

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for output (sklearn 1.0+ compatibility).

        Parameters
        ----------
        input_features : array-like of str or None, optional
            Input feature names. If None, uses feature_names_in_.

        Returns
        -------
        feature_names_out : np.ndarray of str
            Selected feature names.
        """
        check_is_fitted(self, 'selector_')

        if input_features is None:
            input_features = self.feature_names_in_

        if input_features is None:
            # Return indices as strings
            selected_indices = self.get_support(indices=True)
            return np.array([f'x{i}' for i in selected_indices])

        support = self.get_support()
        return np.array(input_features)[support]

    @property
    def selected_features_(self):
        """Get list of selected feature names or indices."""
        check_is_fitted(self, 'selector_')

        if self.feature_names_in_:
            return self.get_feature_names_out().tolist()
        else:
            return self.get_support(indices=True).tolist()

    @property
    def n_features_out_(self):
        """Get number of features selected."""
        check_is_fitted(self, 'selector_')
        return np.sum(self.get_support())

    @property
    def feature_importances_(self):
        """
        Get feature importance scores.

        Returns
        -------
        importances : np.ndarray or dict
            For single method: array of importance scores.
            For multiple methods: dict mapping method names to arrays.
        """
        check_is_fitted(self, 'selector_')

        if isinstance(self.selector_, list):
            # Multiple methods - return dict
            methods = [self.method] if isinstance(self.method, str) else self.method
            return {
                method: selector.get_feature_importances()
                for method, selector in zip(methods, self.selector_)
            }
        else:
            # Single method - return array
            return self.selector_.get_feature_importances()

    def get_report(self) -> pd.DataFrame:
        """
        Generate comprehensive feature selection report.

        Returns
        -------
        report : pd.DataFrame
            Feature-level statistics with columns:
            - feature_name: Feature name or index
            - selected: Whether feature was selected
            - importance_score: Importance score (if available)
            - rank: Feature rank by importance
        """
        check_is_fitted(self, 'selector_')

        feature_names = self.feature_names_in_ if self.feature_names_in_ else \
            [f'feature_{i}' for i in range(self.n_features_in_)]

        support = self.get_support()

        # Get importance scores
        if isinstance(self.selector_, list):
            # Average importances across methods
            importances = self._compute_average_importances()
        else:
            importances = self.feature_importances_
            if importances is None:
                importances = np.ones(self.n_features_in_)

        report = pd.DataFrame({
            'feature_name': feature_names,
            'selected': support,
            'importance_score': importances,
        })

        report['rank'] = report['importance_score'].rank(ascending=False, method='min').astype(int)
        return report.sort_values('rank')
