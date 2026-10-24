"""
Feature selection methods for high-dimensional data, one subpackage each.

Importing a subpackage defines its selector class, which registers it with
FeatureSelector.
"""

from .anova_f import ANOVAFSelector
from .base import FeatureSelector
from .correlation import CorrelationSelector
from .fdr import FDRSelector
from .fwe import FWESelector
from .higher_criticism import HigherCriticismSelector
from .lasso import LassoSelector
from .mutual_info import MutualInfoSelector
from .percentile import PercentileSelector
from .random_selection import RandomSelector
from .rfe import RFESelector
from .rfecv import RFECVSelector
from .sequential import SequentialSelector
from .tree_importance import TreeImportanceSelector
from .variance_threshold import VarianceThreshold

__all__ = [
    'ANOVAFSelector',
    'CorrelationSelector',
    'FDRSelector',
    'FWESelector',
    'FeatureSelector',
    'HigherCriticismSelector',
    'LassoSelector',
    'MutualInfoSelector',
    'PercentileSelector',
    'RFECVSelector',
    'RFESelector',
    'RandomSelector',
    'SequentialSelector',
    'TreeImportanceSelector',
    'VarianceThreshold',
]
