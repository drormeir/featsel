"""
Lasso: rank features by the size of L1-regularized model coefficients.
"""

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from ..base import FeatureSelector


class LassoSelector(FeatureSelector):
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

    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    coef_ : np.ndarray
        Fitted model coefficients.
    feature_importances_ : np.ndarray
        Absolute coefficient per feature.

    Examples
    --------
    >>> from featsel.selectors import LassoSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> LassoSelector(n_features=10, random_state=42).fit_transform(X, y).shape
    (100, 10)
    """

    def __init__(self, n_features=None, C=1.0, alpha=0.01, task='classification',
                 max_iter=1000, random_state=None, check_input=True):
        self.n_features = n_features
        self.C = C
        self.alpha = alpha
        self.task = task
        self.max_iter = max_iter
        self.random_state = random_state
        self.check_input = check_input

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
        X, y = self._validate(X, y)

        if self.task == 'classification':
            model = LogisticRegression(
                l1_ratio=1, C=self.C, solver='liblinear',
                max_iter=self.max_iter, random_state=self.random_state
            )
            # liblinear is the fast L1 solver but is binary only, so multiclass
            # goes through an explicit one-vs-rest wrapper.
            if len(np.unique(y)) > 2:
                model = OneVsRestClassifier(model)
        elif self.task == 'regression':
            model = Lasso(
                alpha=self.alpha, max_iter=self.max_iter,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        model.fit(X, y)
        if isinstance(model, OneVsRestClassifier):
            self.coef_ = np.vstack([est.coef_ for est in model.estimators_])
        else:
            self.coef_ = model.coef_

        # Multiclass gives one coefficient row per class; a feature matters if
        # it matters for any class, so sum the absolute values.
        self.feature_importances_ = np.abs(np.atleast_2d(self.coef_)).sum(axis=0)

        if self.n_features is None:
            self.support_ = self.feature_importances_ != 0
        else:
            self.support_ = self._top(self.feature_importances_)

        if not self.support_.any():
            raise ValueError(
                "L1 regularization zeroed every coefficient. "
                "Increase C (classification) or decrease alpha (regression)."
            )
        return self
