"""
Tree importance: rank features by random forest impurity decrease.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..base import FeatureSelector


class TreeImportanceSelector(FeatureSelector):
    """
    Select features by random forest impurity-based importance.

    Best for: non-linear signal, interactions between features
    Pros: multivariate, captures non-linearity, no scaling needed
    Cons: costs a forest fit, biased towards high-cardinality features,
    splits importance across correlated features

    Parameters
    ----------
    n_features : int
        Number of top features to select. Required at fit time.
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

    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    feature_importances_ : np.ndarray
        Mean impurity decrease per feature.

    Examples
    --------
    >>> from featsel.selectors import TreeImportanceSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> TreeImportanceSelector(n_features=10, random_state=42).fit_transform(X, y).shape
    (100, 10)
    """

    def __init__(self, n_features=None, n_estimators=200, task='classification',
                 max_depth=None, random_state=None, n_jobs=1, check_input=True):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.task = task
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.check_input = check_input

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
        self._require_n_features()
        X, y = self._validate(X, y)

        if self.task == 'classification':
            forest_class = RandomForestClassifier
        elif self.task == 'regression':
            forest_class = RandomForestRegressor
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        forest = forest_class(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, n_jobs=self.n_jobs
        ).fit(X, y)

        self.feature_importances_ = forest.feature_importances_
        self.support_ = self._top(self.feature_importances_)
        return self
