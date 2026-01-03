"""
Input validation utilities for feature selection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def validate_X_y(X, y=None, require_y=False):
    """
    Validate input data and target.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray, optional
        Target values.
    require_y : bool, default=False
        If True, raise error if y is None.

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        Validated feature matrix.
    y : pd.Series or np.ndarray or None
        Validated target values.

    Raises
    ------
    TypeError
        If X is not DataFrame or ndarray.
    ValueError
        If require_y is True and y is None, or if shapes don't match.
    """
    # Validate X
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError(f"X must be pandas DataFrame or numpy array, got {type(X)}")

    if isinstance(X, np.ndarray) and X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

    # Validate y
    if require_y and y is None:
        raise ValueError("Target values (y) are required for this method")

    if y is not None:
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(f"y must be pandas Series or numpy array, got {type(y)}")

        if isinstance(y, np.ndarray) and y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")

        # Check shape compatibility
        n_samples_X = X.shape[0]
        n_samples_y = len(y)
        if n_samples_X != n_samples_y:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {n_samples_X}, y: {n_samples_y}"
            )

    return X, y


def check_n_features(n_features: Optional[int], n_features_total: int):
    """
    Validate n_features parameter.

    Parameters
    ----------
    n_features : int or None
        Requested number of features.
    n_features_total : int
        Total number of features available.

    Raises
    ------
    ValueError
        If n_features is invalid.
    """
    if n_features is None:
        return

    if not isinstance(n_features, int):
        raise TypeError(f"n_features must be int, got {type(n_features)}")

    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    if n_features > n_features_total:
        raise ValueError(
            f"n_features ({n_features}) cannot exceed total features ({n_features_total})"
        )
