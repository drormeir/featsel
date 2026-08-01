"""
Feature selection methods for high-dimensional data.
"""

from .base import BaseSelector
from .baseline import RandomSelector
from .embedded import LassoSelector, TreeImportanceSelector
from .filter import (
    VarianceThreshold,
    ANOVAFSelector,
    MutualInfoSelector,
    CorrelationSelector
)

__all__ = [
    'BaseSelector',
    'RandomSelector',
    'LassoSelector',
    'TreeImportanceSelector',
    'VarianceThreshold',
    'ANOVAFSelector',
    'MutualInfoSelector',
    'CorrelationSelector'
]
