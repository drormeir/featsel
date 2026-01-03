"""
Base class for all feature selection methods.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseSelector(ABC):
    """
    Abstract base class for all feature selection methods.

    All selectors must implement fit() and get_support() methods.
    This ensures a consistent interface across filter, wrapper, and embedded methods.

    Parameters
    ----------
    n_features : int, optional
        Number of features to select. If None, method-specific default is used.
    **kwargs : dict
        Additional method-specific parameters.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the selector has been fitted.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list or None
        Feature names if input was DataFrame, None otherwise.
    """

    def __init__(self, n_features=None, **kwargs):
        self.n_features = n_features
        self.kwargs = kwargs
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.feature_names_in_ = None

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the selector on training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,), optional
            Target values. Required for supervised methods.

        Returns
        -------
        self : BaseSelector
            Fitted selector.
        """
        pass

    @abstractmethod
    def get_support(self, indices=False):
        """
        Get boolean mask or integer indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return integer indices of selected features.
            If False, return boolean mask.

        Returns
        -------
        support : np.ndarray
            Boolean mask (if indices=False) or integer indices (if indices=True)
            of selected features.
        """
        pass

    def transform(self, X):
        """
        Transform X to selected features.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : pd.DataFrame or np.ndarray of shape (n_samples, n_features_out)
            Data with only selected features. Type matches input type.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before transform. Call fit() first.")

        support = self.get_support(indices=False)

        if isinstance(X, pd.DataFrame):
            return X.loc[:, support]
        else:
            return X[:, support]

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,), optional
            Target values.

        Returns
        -------
        X_selected : pd.DataFrame or np.ndarray
            Transformed data with selected features.
        """
        return self.fit(X, y).transform(X)

    def get_feature_importances(self):
        """
        Return feature importance scores if available.

        Returns
        -------
        importances : np.ndarray or None
            Feature importance scores. None if method doesn't provide importance scores.
        """
        return getattr(self, 'feature_importances_', None)

    def _store_feature_info(self, X):
        """
        Store feature information from input data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = len(self.feature_names_in_)
        elif isinstance(X, np.ndarray):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = None
        else:
            raise TypeError(f"X must be pandas DataFrame or numpy array, got {type(X)}")

    def _convert_to_array(self, X):
        """
        Convert input to numpy array if needed.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data.

        Returns
        -------
        X_array : np.ndarray
            Data as numpy array.
        """
        if isinstance(X, pd.DataFrame):
            return X.values
        return X

    def _convert_to_series(self, y):
        """
        Convert target to numpy array if needed.

        Parameters
        ----------
        y : pd.Series or np.ndarray or None
            Target values.

        Returns
        -------
        y_array : np.ndarray or None
            Target as numpy array, or None if y is None.
        """
        if y is None:
            return None
        if isinstance(y, pd.Series):
            return y.values
        return y
