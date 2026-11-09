"""
The base class of every feature selector.
"""

from __future__ import annotations

import numbers
import random
import sys
from abc import abstractmethod
from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class FeatureSelector(SelectorMixin, BaseEstimator):
    """
    Base class of every feature selector: a scikit-learn estimator.

    A subclass lists its parameters explicitly in __init__, only stores them,
    and implements _fit(), which sets `support_`: a boolean mask over the input
    features. scikit-learn's SelectorMixin then provides transform(),
    fit_transform(), get_support() and get_feature_names_out(), and
    BaseEstimator provides get_params() and set_params(), so selectors work
    with Pipeline, clone(), cross_val_score() and GridSearchCV.

    As in scikit-learn, transform() returns a NumPy array. Call
    set_output(transform='pandas') to keep DataFrame columns. Given a PyTorch
    tensor, transform() returns a tensor on the same device.

    X may be a PyTorch tensor. A selector with a GPU path keeps a tensor on
    a GPU there, with no copy to the CPU; every other selector, and every
    tensor on the CPU, goes through NumPy. y may be a tensor too.

    Every subclass takes check_input=True. False skips validating X and y in
    fit, as scikit-learn's own check_input does. Use it only when X is already
    a finite 2-D NumPy array and y a 1-D array.

    A subclass that uses randomness derives from FeatureSelectorRandom.

    A subclass keeps its per-feature measurement under the name of what it
    measures, as scikit-learn does: scores_, log_pvalues_, coef_, variances_.
    A subclass that orders the features also defines ranking(): feature
    indices, best first, whose first k are exactly what a fit with
    n_features=k selects. A subclass with no such order has no ranking().

    Every subclass registers itself when it is defined, under its class name
    and any aliases given as a class keyword:

        class MySelector(FeatureSelector, aliases=['mine', 'my_sel']): ...

    In Python, import the class. FeatureSelector.create(name, **params) is for
    names that arrive as text, such as a YAML config; lookup ignores case. The
    base class never imports its subclasses.
    """

    # lowercased name or alias -> selector class
    _registry: ClassVar[dict[str, type[FeatureSelector]]] = {}

    # True when the selector has ranking() and its first k, fit with any
    # n_features <= k, are the selection a fit with n_features=k would make.
    # The runner then fits once per split and cuts the ranking at each k.
    rank_once: ClassVar[bool] = False

    # Set by fit: support_ by the subclass's _fit, n_features_in_ by _validate.
    support_: np.ndarray
    n_features_in_: int

    def __init_subclass__(cls, aliases: str | list[str] | None = None, register: bool = True,
                          **kwargs):
        super().__init_subclass__(**kwargs)
        if not register:
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

    def fit(self, X, y=None):
        """Learn which features to keep and return self."""
        return self._fit(X, y)

    @abstractmethod
    def _fit(self, X, y=None):
        """Learn which features to keep; set `support_` and return self."""

    def transform(self, X):
        # Check fitting first: SelectorMixin would otherwise validate X against
        # an unfitted selector and warn about feature names before failing.
        check_is_fitted(self, 'support_')
        torch = sys.modules.get('torch')   # a tensor exists only once PyTorch is imported
        if torch is not None and isinstance(X, torch.Tensor):
            if X.ndim != 2 or X.shape[1] != self.n_features_in_:
                raise ValueError(f"X has shape {tuple(X.shape)}, but {type(self).__name__} "
                                 f"was fitted on {self.n_features_in_} features")
            return X[:, torch.as_tensor(np.flatnonzero(self.support_), device=X.device)]
        return super().transform(X)

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

    def _validate(self, X, y=None, requires_y=True, keep_gpu_tensor=False, *, check_input):
        """
        Validate the training data and record the input feature count and names.

        Sets n_features_in_, and feature_names_in_ for a DataFrame. With
        check_input=False only the feature count is recorded, and X and y are
        returned as given.

        A PyTorch tensor y becomes a NumPy array. A tensor X on a GPU is
        returned as it is with keep_gpu_tensor, checked on its device, and
        copied to a NumPy array otherwise; a tensor X on the CPU becomes a
        NumPy array without a copy.
        """
        if y is None and requires_y:
            raise ValueError(f"{type(self).__name__} requires target values (y)")
        torch = sys.modules.get('torch')
        if torch is not None:
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            if isinstance(X, torch.Tensor):
                if keep_gpu_tensor and X.device.type != 'cpu':
                    return self._validate_gpu_tensor(torch, X, y, check_input), y
                X = X.detach().cpu().numpy()
        if not check_input:
            self.n_features_in_ = X.shape[1]
            return X, y
        if y is None:
            return validate_data(self, X), None
        return validate_data(self, X, y)

    def _validate_gpu_tensor(self, torch, X, y, check_input):
        """Check a tensor on a GPU where it is, as validate_data checks an array."""
        if X.ndim != 2:
            raise ValueError(f"Expected a 2-D tensor, got shape {tuple(X.shape)}")
        if y is not None and len(y) != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)}")
        if check_input and not torch.isfinite(X).all():
            raise ValueError("Input X contains NaN or infinity.")
        self.n_features_in_ = X.shape[1]
        self.__dict__.pop('feature_names_in_', None)   # a refit on a tensor has no names
        return X

    @staticmethod
    def _best_first(scores):
        """Feature indices by score, largest first."""
        return np.argsort(scores)[::-1]

    @staticmethod
    def _by_task(task, classification, regression):
        """Whichever of the two matches task."""
        if task == 'classification':
            return classification
        if task == 'regression':
            return regression
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")

    def _mask(self, indices):
        """A boolean mask over the input features with `indices` set."""
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[indices] = True
        return mask


class FeatureSelectorRandom(FeatureSelector, register=False):
    """
    Base class of every selector that uses randomness.

    A subclass takes random_state, default 42, so a fit repeats without a seed
    being passed, and calls super().__init__(random_state). It passes the seed
    to scikit-learn, or draws from _rng(), a generator built afresh on every
    fit.

    Every fit first seeds the shared generators of Python, NumPy and, once
    imported, PyTorch (its CUDA generators once CUDA has started). So
    randomness drawn from shared state, here or in a library, repeats too.
    random_state must be an integer.
    """

    def __init__(self, random_state=42):
        assert isinstance(random_state, numbers.Integral), \
            f"random_state must be an integer, got {random_state!r}"
        self.random_state = random_state

    def fit(self, X, y=None):
        """Seed the shared random generators, then learn which features to keep."""
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch = sys.modules.get('torch')
        if torch is not None:
            # The CPU and CUDA generators directly: torch.manual_seed also
            # visits every other backend, at about 0.3 ms per fit. CUDA
            # only once started: before, the call is queued with a stack
            # trace, at about 0.1 ms per fit.
            torch.default_generator.manual_seed(self.random_state)
            if torch.cuda.is_initialized():
                torch.cuda.manual_seed_all(self.random_state)
        return super().fit(X, y)

    def _rng(self):
        """
        A fresh generator seeded by self.random_state.

        Built anew on every fit, so refitting repeats the same draws whatever
        else ran before.
        """
        return np.random.default_rng(self.random_state)
