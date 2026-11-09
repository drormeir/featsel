"""
Does RFECV's inner score decide how many genes it keeps on SCAN-B?

RFECV picks its gene count by inner cross-validation. Its default inner
score is accuracy. On the frontier (docs/final_report/08_results_frontier.md)
it kept 1 or 2 genes on the imbalanced subtype tasks. This script takes one
70/30 stratified split, fits RFECV with three inner scores on four tasks,
and reports the genes kept and the test AUC of logistic regression on them.

Usage:
    python scripts/rfecv_scoring.py
"""

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from featsel.selectors import RFECVSelector
from featsel_datasets.scanb_gse96058.featsel_view import load_xy

SCORINGS = (None, 'roc_auc', 'balanced_accuracy')
TASKS = ('Normal', 'Her2', 'LumB', 'Basal')


def main():
    X, y = load_xy()
    X = X.to_numpy(np.float32)
    labels = y.to_numpy()
    train, test = next(StratifiedShuffleSplit(1, train_size=0.7, random_state=42).split(X, labels))
    scaler = StandardScaler().fit(X[train])
    X_train, X_test = scaler.transform(X[train]), scaler.transform(X[test])

    for subtype in TASKS:
        y_train = (labels[train] == subtype).astype(int)
        y_test = (labels[test] == subtype).astype(int)
        for scoring in SCORINGS:
            start = time.perf_counter()
            selector = RFECVSelector(C=0.01, cv=5, solver='newton', device='cpu',
                                     scoring=scoring).fit(X_train, y_train)
            seconds = time.perf_counter() - start
            genes = selector.get_support(indices=True)
            model = LogisticRegression(max_iter=1000).fit(X_train[:, genes], y_train)
            auc = roc_auc_score(y_test, model.decision_function(X_test[:, genes]))
            print(f'{subtype:7s} scoring={scoring!s:18s} genes={len(genes):6d} '
                  f'test AUC={auc:.3f} ({seconds:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
