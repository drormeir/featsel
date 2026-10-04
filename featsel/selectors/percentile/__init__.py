"""
Percentile: keep the top percentage of features by F statistic.
"""

from sklearn.feature_selection import SelectPercentile, f_classif, f_regression

from ..base import FeatureSelector


class PercentileSelector(FeatureSelector):
    """
    Select the top percentage of features by ANOVA F statistic.

    The same ranking as ANOVAFSelector, but the count is a share of the input
    features instead of a fixed number. Useful when datasets of different
    widths should keep the same share.

    Wraps scikit-learn's SelectPercentile.

    Best for: comparing datasets with different feature counts
    Pros: fast, scale-free cutoff
    Cons: same blind spots as the F-test; the count grows with the input
    even when the signal does not

    Parameters
    ----------
    percentile : float, default=10
        Percentage of features to keep, from 0 to 100.
    task : str, default='classification'
        'classification' uses f_classif, 'regression' uses f_regression.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    scores_ : np.ndarray
        F statistic for each feature.
    pvalues_ : np.ndarray
        p-value for each feature.
    feature_importances_ : np.ndarray
        Alias for scores_.

    Examples
    --------
    >>> from featsel.selectors import PercentileSelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=40, n_informative=10)
    >>> PercentileSelector(percentile=25).fit_transform(X, y).shape
    (100, 10)
    """

    def __init__(self, percentile=10, task='classification', check_input=True):
        self.percentile = percentile
        self.task = task
        self.check_input = check_input

    def fit(self, X, y):
        """
        Keep the highest-scoring percentile of features.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : PercentileSelector
            Fitted selector.
        """
        X, y = self._validate(X, y)

        if self.task == 'classification':
            score_func = f_classif
        elif self.task == 'regression':
            score_func = f_regression
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{self.task}'")

        top = SelectPercentile(score_func, percentile=self.percentile).fit(X, y)
        self.scores_ = top.scores_
        self.pvalues_ = top.pvalues_
        self.feature_importances_ = self.scores_
        self.support_ = top.get_support()
        return self
