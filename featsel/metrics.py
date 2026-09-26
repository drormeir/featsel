"""
The runner's results format: the columns that identify an execution and the
predictive metrics recorded for it.

Kept apart from `run.py` so that `analysis.py`, and through it the package
root, can use them without importing the runner.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
)


def g_mean(y_true, y_pred):
    """
    Geometric mean of per-class recalls (Kubat and Matwin, 1997).

    Stricter than macro-F1 on imbalanced targets: it collapses to zero if any
    single class is never predicted correctly.
    """
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    if np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


# name -> callable(y_true, y_pred) -> float. Add a metric with one line.
METRICS = {
    'accuracy': accuracy_score,
    'macro_f1': lambda t, p: f1_score(t, p, average='macro', zero_division=0),
    'balanced_accuracy': balanced_accuracy_score,
    'mcc': matthews_corrcoef,
    'g_mean': g_mean,
}

# Columns that identify an execution. Used to skip finished work on --resume.
KEY_COLUMNS = ['task', 'split', 'train_size', 'preprocess', 'selector', 'k', 'model']
