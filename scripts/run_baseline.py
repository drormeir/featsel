"""One end-to-end baseline run on SCAN-B.

Random selection (the control) and ANOVA F, both at top 100 genes, each
followed by logistic regression under stratified 5-fold CV. Every task is
binary (SCOPE.md 3a): each PAM50 subtype against the rest. Prints accuracy and
macro-F1 per subtype and method.

Feature selection sits inside the pipeline, so it is fit on the training
split of each fold only (SCOPE.md 3a: selection inside the fold).

Data: GSE96058 through `load_xy`, one HiSeq 2000 sample per patient.

Usage: python scripts/run_baseline.py
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from featsel.selectors import ANOVAFSelector, RandomSelector
from featsel_datasets.scanb_gse96058.featsel_view import load_xy

N_FEATURES = 100
N_SPLITS = 5
SEED = 42
SELECTORS = [
    RandomSelector(n_features=N_FEATURES, random_state=SEED),
    ANOVAFSelector(n_features=N_FEATURES),
]


def main() -> None:
    X, y = load_xy(target="pam50 subtype")

    print(f"\nClass distribution:\n{y.value_counts().to_string()}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for subtype, selector in product(sorted(y.unique()), SELECTORS):
        y_task = np.where(y == subtype, subtype, "rest")
        method = type(selector).__name__
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", selector),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])

        scores = cross_validate(
            pipe, X, y_task, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1
        )
        acc, f1 = scores["test_accuracy"], scores["test_f1_macro"]
        fit_time = scores["fit_time"]

        print(f"\n=== {subtype} vs rest, {N_SPLITS}-fold stratified CV, "
              f"{method} top-{N_FEATURES} ===")
        print(f"Accuracy : {acc.mean():.4f} +/- {acc.std():.4f}  {np.round(acc, 4)}")
        print(f"Macro-F1 : {f1.mean():.4f} +/- {f1.std():.4f}  {np.round(f1, 4)}")
        print(f"Fit time : {fit_time.mean():.1f}s per fold")


if __name__ == "__main__":
    main()
