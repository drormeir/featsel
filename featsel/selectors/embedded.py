"""
Embedded feature selection methods.

Embedded methods select features as a side effect of fitting a model, so the
selection reflects how features work together rather than one at a time. They
cost a full model fit per selection, which is the trade-off against filters.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .base import BaseSelector


class LassoSelector(BaseSelector):
    """
    Select features by the magnitude of L1-regularized model coefficients.

    Best for: sparse linear signal, correlated features that should be pruned
    Pros: multivariate, ranks and sparsifies in one fit, well understood theory
    Cons: costs a model fit, unstable when features are strongly correlated,
    picks roughly one feature per correlated group

    For classification this is L1-penalized logistic regression, for regression
    it is Lasso. Feature importance is the absolute coefficient, summed over
    classes in the multiclass case.

    Parameters
    ----------
    n_features : int, optional
        Number of top features to select by absolute coefficient. If None, all
        features with a non-zero coefficient are selected, which lets the
        method choose its own feature count.
    C : float, default=1.0
        Inverse regularization strength for classification. Smaller is sparser.
    alpha : float, default=0.01
        Regularization strength for regression. Larger is sparser.
    task : str, default='classification'
        Type of task: 'classification' or 'regression'.
    max_iter : int, default=1000
        Maximum solver iterations.
    random_state : int, optional
        Random seed for the solver.
    **kwargs : dict
        Additional arguments (unused).

    Attributes
    ----------
    coef_ : np.ndarray
        Fitted model coefficients.
    feature_importances_ : np.ndarray
        Absolute coefficient per feature.

    Examples
    --------
    >>> from featsel.selectors.embedded import LassoSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> selector = LassoSelector(n_features=10, random_state=42)
    >>> selector.fit(X, y)
    >>> selector.transform(X).shape
    (100, 10)
    """

    def __init__(self, n_features=None, C=1.0, alpha=0.01, task='classification',
                 max_iter=1000, random_state=None, **kwargs):
        super().__init__(n_features=n_features, **kwargs)
        self.C = C
        self.alpha = alpha
        self.task = task
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the L1 model and rank features by absolute coefficient.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LassoSelector
            Fitted selector.
        """
        if y is None:
            raise ValueError("LassoSelector requires target values (y)")

        self._store_feature_info(X)
        X_array = self._convert_to_array(X)
        y_array = self._convert_to_series(y)

        if self.task == 'classification':
            model = LogisticRegression(
                l1_ratio=1, C=self.C, solver='liblinear',
                max_iter=self.max_iter, random_state=self.random_state
            )
            # liblinear is the fast L1 solver but is binary only, so multiclass
            # goes through an explicit one-vs-rest wrapper.
            if len(np.unique(y_array)) > 2:
                model = OneVsRestClassifier(model)
        elif self.task == 'regression':
            model = Lasso(
                alpha=self.alpha, max_iter=self.max_iter,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        model.fit(X_array, y_array)
        if isinstance(model, OneVsRestClassifier):
            self.coef_ = np.vstack([est.coef_ for est in model.estimators_])
        else:
            self.coef_ = model.coef_

        # Multiclass gives one coefficient row per class; a feature matters if
        # it matters for any class, so sum the absolute values.
        coef = np.atleast_2d(self.coef_)
        self.feature_importances_ = np.abs(coef).sum(axis=0)

        nonzero = np.flatnonzero(self.feature_importances_)
        if self.n_features is None:
            self.selected_indices_ = nonzero
        else:
            ranked = np.argsort(self.feature_importances_)[::-1]
            self.selected_indices_ = np.sort(ranked[:self.n_features])

        if len(self.selected_indices_) == 0:
            raise ValueError(
                "L1 regularization zeroed every coefficient. "
                f"Increase C (classification) or decrease alpha (regression)."
            )

        self.is_fitted_ = True
        return self

    def get_support(self, indices=False):
        """
        Get boolean mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return integer indices.
            If False, return boolean mask.

        Returns
        -------
        support : np.ndarray
            Boolean mask or integer indices of selected features.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before get_support")

        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_indices_] = True
        return self.selected_indices_ if indices else support


class TreeImportanceSelector(BaseSelector):
    """
    Select features by random forest impurity-based importance.

    Best for: non-linear signal, interactions between features
    Pros: multivariate, captures non-linearity, no scaling needed
    Cons: costs a forest fit, biased towards high-cardinality features,
    splits importance across correlated features

    Parameters
    ----------
    n_features : int
        Number of top features to select.
    n_estimators : int, default=200
        Number of trees in the forest. More trees give a more stable ranking.
    task : str, default='classification'
        Type of task: 'classification' or 'regression'.
    max_depth : int, optional
        Maximum tree depth. None grows trees fully.
    random_state : int, optional
        Random seed for the forest.
    n_jobs : int, default=1
        Threads used to fit the forest.
    **kwargs : dict
        Additional arguments (unused).

    Attributes
    ----------
    feature_importances_ : np.ndarray
        Mean impurity decrease per feature.

    Examples
    --------
    >>> from featsel.selectors.embedded import TreeImportanceSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> selector = TreeImportanceSelector(n_features=10, random_state=42)
    >>> selector.fit(X, y)
    >>> selector.transform(X).shape
    (100, 10)
    """

    def __init__(self, n_features, n_estimators=200, task='classification',
                 max_depth=None, random_state=None, n_jobs=1, **kwargs):
        super().__init__(n_features=n_features, **kwargs)
        self.n_estimators = n_estimators
        self.task = task
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Fit the forest and rank features by impurity decrease.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : TreeImportanceSelector
            Fitted selector.
        """
        if y is None:
            raise ValueError("TreeImportanceSelector requires target values (y)")
        if self.n_features is None:
            raise ValueError("TreeImportanceSelector requires n_features to be specified")

        self._store_feature_info(X)
        X_array = self._convert_to_array(X)
        y_array = self._convert_to_series(y)

        if self.task == 'classification':
            forest_class = RandomForestClassifier
        elif self.task == 'regression':
            forest_class = RandomForestRegressor
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        forest = forest_class(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        forest.fit(X_array, y_array)

        self.feature_importances_ = forest.feature_importances_
        ranked = np.argsort(self.feature_importances_)[::-1]
        self.selected_indices_ = np.sort(ranked[:self.n_features])

        self.is_fitted_ = True
        return self

    def get_support(self, indices=False):
        """
        Get boolean mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return integer indices.
            If False, return boolean mask.

        Returns
        -------
        support : np.ndarray
            Boolean mask or integer indices of selected features.
        """
        if not self.is_fitted_:
            raise RuntimeError("Selector must be fitted before get_support")

        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_indices_] = True
        return self.selected_indices_ if indices else support
