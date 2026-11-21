"""
Simulated data with rare, weak signal, where the informative features are known.

Higher Criticism and Berk-Jones are built for signal that is rare (a small
share of the features), weak (each feature alone is near the detection
limit) and independent. None of the three real datasets has such signal.
This simulation has it, so it tests whether these thresholds choose a good
count where their assumptions hold.

The model:

- Every feature is independent and standard normal in both classes.
- In n_signal features, chosen at random positions, the positive class
  has its mean moved by `shift` (in standard deviations, Cohen's d).

An experiment config names this module as its `dataset`; `dataset_params`
change the settings. The runner reads `truth(X)` and records, for each
execution, the share of the informative features selected (truth_recall)
and the share of the selection that is informative (truth_precision).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = 'Simulated rare signal'
TARGET = 'label'


def load_xy(
    target: str = TARGET,
    n_samples: int = 500,
    n_features: int = 10_000,
    n_signal: int = 40,
    shift: float = 0.4,
    positive_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return X (samples x features) and y, 'positive' or 'negative'.

    An informative feature is named 'signal_<position>', any other
    'null_<position>'. The same arguments give the same data.
    """
    rng = np.random.default_rng(seed)
    n_positive = round(n_samples * positive_fraction)
    positive = np.zeros(n_samples, dtype=bool)
    positive[rng.choice(n_samples, n_positive, replace=False)] = True

    X = rng.standard_normal((n_samples, n_features))
    signal = rng.choice(n_features, n_signal, replace=False)
    X[np.ix_(positive, signal)] += shift

    names = np.array([f'null_{i:05d}' for i in range(n_features)], dtype=object)
    names[signal] = [f'signal_{i:05d}' for i in signal]
    index = pd.Index([f'sample_{i:04d}' for i in range(n_samples)], name='sample')
    y = pd.Series(np.where(positive, 'positive', 'negative'), index=index, name=target)
    return pd.DataFrame(X, index=index, columns=names), y


def truth(X: pd.DataFrame) -> np.ndarray:
    """The column indices of the informative features in X."""
    return np.flatnonzero(X.columns.str.startswith('signal_'))
