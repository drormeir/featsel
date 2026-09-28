"""
Feature selection methods for high-dimensional data, one subpackage each.

Importing a subpackage defines its selector class, which registers it with
FeatureSelector.
"""

from .anova_f import ANOVAFSelector
from .base import FeatureSelector
from .correlation import CorrelationSelector
from .lasso import LassoSelector
from .mutual_info import MutualInfoSelector
from .random_selection import RandomSelector
from .tree_importance import TreeImportanceSelector
from .variance_threshold import VarianceThreshold

__all__ = [
    'ANOVAFSelector',
    'CorrelationSelector',
    'FeatureSelector',
    'LassoSelector',
    'MutualInfoSelector',
    'RandomSelector',
    'TreeImportanceSelector',
    'VarianceThreshold',
]
