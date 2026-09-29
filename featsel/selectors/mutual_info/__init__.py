"""
Mutual information: rank features by their dependence on the target.
"""

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from ..base import FeatureSelector


class MutualInfoSelector(FeatureSelector):
    """
    Select features based on mutual information with target.

    Best for: Capturing non-linear relationships, both classification and regression
    Pros: Detects any relationship (linear/non-linear), model-agnostic
    Cons: Computationally expensive (O(n*m*log(n))), requires hyperparameter tuning

    Parameters
    ----------
    n_features : int
        Number of top features to select. Required at fit time.
    task : str, default='classification'
        Type of task: 'classification' or 'regression'.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
        Lower values = more sensitive to local structure.
    random_state : int, optional
        Random seed for reproducibility.

    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    feature_importances_ : np.ndarray
        Mutual information scores for each feature.

    Examples
    --------
    >>> from featsel.selectors import MutualInfoSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> MutualInfoSelector(n_features=10, random_state=42).fit_transform(X, y).shape
    (100, 10)
    """

    def __init__(self, n_features=None, task='classification', n_neighbors=3,
                 random_state=None, check_input=True):
        self.n_features = n_features
        self.task = task
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.check_input = check_input

    def fit(self, X, y):
        """
        Fit the selector on training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : MutualInfoSelector
            Fitted selector.
        """
        self._require_n_features()
        X, y = self._validate(X, y)

        if self.task == 'classification':
            mi_func = mutual_info_classif
        elif self.task == 'regression':
            mi_func = mutual_info_regression
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        self.feature_importances_ = mi_func(
            X, y, n_neighbors=self.n_neighbors, random_state=self.random_state
        )
        self.support_ = self._top(self.feature_importances_)
        return self
