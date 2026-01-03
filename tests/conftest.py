"""
Shared test fixtures for featsel tests.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def small_classification_data():
    """
    Small synthetic classification dataset for fast unit tests.

    Returns
    -------
    X : pd.DataFrame of shape (100, 20)
        Feature matrix with 10 informative features.
    y : pd.Series of shape (100,)
        Binary target variable.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        shuffle=False
    )
    feature_names = [f'feature_{i}' for i in range(20)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    return X_df, y_series


@pytest.fixture
def small_regression_data():
    """
    Small synthetic regression dataset for fast unit tests.

    Returns
    -------
    X : pd.DataFrame of shape (100, 20)
        Feature matrix with 10 informative features.
    y : pd.Series of shape (100,)
        Continuous target variable.
    """
    X, y = make_regression(
        n_samples=100,
        n_features=20,
        n_informative=10,
        random_state=42,
        shuffle=False
    )
    feature_names = [f'feature_{i}' for i in range(20)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    return X_df, y_series


@pytest.fixture
def high_dim_data():
    """
    High-dimensional data (n_features >> n_samples) for realistic testing.

    This simulates the user's use case: transfer learning with pre-extracted
    embeddings where embedding dimension greatly exceeds sample count.

    Returns
    -------
    X : pd.DataFrame of shape (50, 500)
        Feature matrix with 50 informative features.
    y : pd.Series of shape (50,)
        Multiclass target variable.
    """
    X, y = make_classification(
        n_samples=50,
        n_features=500,
        n_informative=50,
        n_redundant=50,
        n_classes=3,
        random_state=42,
        shuffle=False
    )
    feature_names = [f'embedding_{i}' for i in range(500)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='class_label')
    return X_df, y_series


@pytest.fixture
def multiclass_data():
    """
    Multiclass classification dataset (5 classes).

    Returns
    -------
    X : pd.DataFrame of shape (150, 30)
        Feature matrix.
    y : pd.Series of shape (150,)
        Target with 5 classes.
    """
    X, y = make_classification(
        n_samples=150,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42,
        shuffle=False
    )
    feature_names = [f'feature_{i}' for i in range(30)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    return X_df, y_series


@pytest.fixture
def data_with_constant_features():
    """
    Dataset with some constant and near-constant features for testing variance threshold.

    Returns
    -------
    X : pd.DataFrame of shape (100, 15)
        Feature matrix with 3 constant features.
    y : pd.Series of shape (100,)
        Target variable.
    """
    np.random.seed(42)
    X_normal = np.random.randn(100, 10)
    X_constant = np.zeros((100, 3))
    X_low_var = np.random.randn(100, 2) * 0.01
    X = np.hstack([X_normal, X_constant, X_low_var])

    feature_names = [f'normal_{i}' for i in range(10)] + \
                   [f'constant_{i}' for i in range(3)] + \
                   [f'low_var_{i}' for i in range(2)]

    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(np.random.randint(0, 2, 100), name='target')
    return X_df, y_series


@pytest.fixture
def scanb_loader():
    """
    Real SCAN-B dataset loader (if data available).

    Returns
    -------
    loader : DataLoader
        Loaded SCAN-B dataset.

    Notes
    -----
    Skips test if SCAN-B data is not available.
    """
    from featsel import DataLoader
    try:
        return DataLoader('configs/scanb.yaml')
    except FileNotFoundError:
        pytest.skip("SCAN-B dataset not available")
