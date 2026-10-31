"""
Mann-Whitney U: rank features by their two-sided U-test p-value.
"""

from ..base import FeatureSelector
from ..log_pvalue import log_mannwhitneyu


class MannWhitneySelector(FeatureSelector):
    """
    Select the n_features with the smallest Mann-Whitney U p-values.

    The rank-based counterpart of ANOVAFSelector: the same top-k rule, with
    the F-test replaced by the U test. With two classes the U test is the
    same test as Kruskal-Wallis, "ANOVA on ranks".

    The U test uses only the order of each feature's values: its U is the
    feature's AUC times n1 n2, and its p-value is the chance that a random
    assignment of the labels, with the class sizes fixed, ranks the classes at
    least as well. Any monotone transform of a feature leaves the selection
    unchanged. Ties get average ranks; the p-value is the tie-corrected normal
    approximation, kept as a log so it stays finite below 1e-308.

    Best for: features whose scale is arbitrary, or far from normal, such as
    log expression with many values tied at a floor
    Pros: no normality assumption; invariant to monotone transforms
    Cons: one feature at a time, so correlated features all pass together;
    two classes only; ranking costs about 10 s on 1,500 x 28,500

    Parameters
    ----------
    n_features : int
        Number of features to keep.
    check_input : bool, default=True
        Validate X and y in fit. See FeatureSelector.

    Attributes
    ----------
    auc_ : np.ndarray
        AUC of each feature for the second class in sorted order.
    log_pvalues_ : np.ndarray
        Natural log of each feature's p-value. A constant feature gets 0.
    feature_importances_ : np.ndarray
        -log_pvalues_: larger is stronger.

    Examples
    --------
    >>> from featsel.selectors import MannWhitneySelector
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_features=50, n_informative=5,
    ...                            n_redundant=0, shuffle=False, random_state=0)
    >>> # Features 0-4 are informative.
    >>> sorted(MannWhitneySelector(n_features=4).fit(X, y).get_support(indices=True).tolist())
    [0, 1, 2, 4]
    """

    rank_once = True

    def __init__(self, n_features=None, check_input=True):
        self.n_features = n_features
        self.check_input = check_input

    def fit(self, X, y):
        """
        Keep the n_features with the smallest U-test p-values.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data.
        y : pd.Series or np.ndarray of shape (n_samples,)
            Class labels, exactly two classes.

        Returns
        -------
        self : MannWhitneySelector
            Fitted selector.
        """
        self._require_n_features()
        X, y = self._validate(X, y)

        self.log_pvalues_, self.auc_ = log_mannwhitneyu(X, y)
        self.feature_importances_ = -self.log_pvalues_
        self.support_ = self._top()
        return self
