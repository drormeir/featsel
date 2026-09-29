"""
The base class of every feature selector.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


def _has_abstract_methods(cls):
    """
    Whether cls still leaves an abstract method unimplemented.

    ABCMeta sets __abstractmethods__ only after __init_subclass__ has run, so
    registration has to inspect the methods itself.
    """
    return any(getattr(getattr(cls, name, None), '__isabstractmethod__', False)
               for name in dir(cls))


class FeatureSelector(SelectorMixin, BaseEstimator):
    """
    Base class of every feature selector: a scikit-learn estimator.

    A subclass lists its parameters explicitly in __init__, only stores them,
    and implements fit(), which sets `support_`: a boolean mask over the input
    features. scikit-learn's SelectorMixin then provides transform(),
    fit_transform(), get_support() and get_feature_names_out(), and
    BaseEstimator provides get_params() and set_params(), so selectors work
    with Pipeline, clone(), cross_val_score() and GridSearchCV.

    As in scikit-learn, transform() returns a NumPy array. Call
    set_output(transform='pandas') to keep DataFrame columns.

    Every subclass takes check_input=True. False skips validating X and y in
    fit, as scikit-learn's own check_input does. Use it only when X is already
    a finite 2-D NumPy array and y a 1-D array.

    Every concrete subclass registers itself when it is defined, under its
    class name and any aliases given as a class keyword:

        class ANOVAFSelector(FeatureSelector, aliases=['anova', 'anova_f']): ...

    In Python, import the class. FeatureSelector.create(name, **params) is for
    names that arrive as text, such as a YAML config; lookup ignores case. The
    base class never imports its subclasses.
    """

    # lowercased name or alias -> concrete selector class
    _registry: ClassVar[dict[str, type[FeatureSelector]]] = {}

    def __init_subclass__(cls, aliases: str | list[str] | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if _has_abstract_methods(cls):
            return

        names = [cls.__name__, *([aliases] if isinstance(aliases, str) else aliases or [])]
        for name in {n.lower() for n in names}:
            owner = FeatureSelector._registry.setdefault(name, cls)
            if owner is not cls:
                raise ValueError(f"Selector name '{name}' of {cls.__name__} is already "
                                 f"taken by {owner.__name__}")

    @classmethod
    def lookup(cls, name: str) -> type[FeatureSelector]:
        """The selector class registered under name, case-insensitively."""
        selector_cls = FeatureSelector._registry.get(name.lower())
        if selector_cls is None:
            available = ', '.join(sorted(FeatureSelector._registry))
            raise ValueError(f"Unknown selector '{name}'. Available: {available}")
        return selector_cls

    @classmethod
    def create(cls, name: str, **params) -> FeatureSelector:
        """Build the selector registered under name. A misspelled parameter raises."""
        return cls.lookup(name)(**params)

    @abstractmethod
    def fit(self, X, y=None):
        """Learn which features to keep; set `support_` and return self."""

    def transform(self, X):
        # Check fitting first: SelectorMixin would otherwise validate X against
        # an unfitted selector and warn about feature names before failing.
        check_is_fitted(self, 'support_')
        return super().transform(X)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

    def _validate(self, X, y=None, requires_y=True):
        """
        Validate the training data and record the input feature count and names.

        With check_input=False only the feature count is recorded, and X and y
        are returned as given.
        """
        if y is None and requires_y:
            raise ValueError(f"{type(self).__name__} requires target values (y)")
        if not self.check_input:
            self.n_features_in_ = X.shape[1]
            return X, y
        if y is None:
            return validate_data(self, X), None
        return validate_data(self, X, y)

    def _require_n_features(self):
        if self.n_features is None:
            raise ValueError(f"{type(self).__name__} requires n_features to be specified")

    def _mask(self, indices):
        """A boolean mask over the input features with `indices` set."""
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[indices] = True
        return mask

    def _top(self, scores):
        """A mask selecting the n_features highest scores."""
        return self._mask(np.argsort(scores)[::-1][:self.n_features])
