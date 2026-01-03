"""
Feature selection methods for high-dimensional data.
"""

from .base import BaseSelector
from .filter import (
    VarianceThreshold,
    ANOVAFSelector,
    MutualInfoSelector,
    CorrelationSelector
)

__all__ = [
    'BaseSelector',
    'VarianceThreshold',
    'ANOVAFSelector',
    'MutualInfoSelector',
    'CorrelationSelector'
]
