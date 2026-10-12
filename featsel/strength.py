"""
Leave-one-out confusion matrix of L2 logistic regression on chosen features.

The design is in `docs/selection_strength.md`. An outer loop holds out one row
at a time. On the rest, an inner leave-one-out loop picks C by balanced
log-loss, and the model refit with that C predicts the held-out row. To keep
runs fast, each loop visits only a random fraction `alpha` of its rows.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CS = np.logspace(-4, 4, 9)


def _model(C):
    # lbfgs leaves the bias unpenalized; genes are standardized on the
    # training rows so the penalty treats them alike.
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=1000)
    )


def _n_rows(alpha, n):
    return max(int(alpha * n), 1)


def loo_confusion(X, y, alpha=0.01, Cs=DEFAULT_CS, random_state=0):
    """
    Confusion matrix of the outer leave-one-out predictions.

    X : (n, k) the features already chosen. y : n binary labels.
    alpha : fraction of rows each loop holds out, one at a time, drawn without
        replacement. The outer loop visits max(alpha*n, 1) rows, the inner
        loop max(alpha*(n-1), 1).
    Cs : grid of inverse penalties, C = 1/lambda.

    Returns the 2x2 matrix, rows true and columns predicted, in the order of
    np.unique(y).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    classes = np.unique(y)
    rng = np.random.default_rng(random_state)

    y_true, y_pred = [], []
    for i in rng.choice(n, _n_rows(alpha, n), replace=False):
        train = np.delete(np.arange(n), i)
        # Balanced log-loss weights, counted on this round's training rows.
        counts = dict(zip(*np.unique(y[train], return_counts=True)))
        weight = {c: len(train) / (2 * counts[c]) for c in classes}

        inner = rng.choice(train, _n_rows(alpha, len(train)), replace=False)
        losses = np.zeros(len(Cs))
        for j in inner:
            fit_rows = train[train != j]
            for c_idx, C in enumerate(Cs):
                model = _model(C).fit(X[fit_rows], y[fit_rows])
                # -log p(true label) from the margin; predict_log_proba gives
                # -inf once the probability rounds to zero.
                margin = model.decision_function(X[j : j + 1])[0]
                sign = 1 if y[j] == model.classes_[1] else -1
                losses[c_idx] += weight[y[j]] * np.logaddexp(0, -sign * margin)

        best = _model(Cs[np.argmin(losses)]).fit(X[train], y[train])
        y_true.append(y[i])
        y_pred.append(best.predict(X[i : i + 1])[0])

    return confusion_matrix(y_true, y_pred, labels=classes)
