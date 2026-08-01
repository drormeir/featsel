"""One end-to-end baseline run on SCAN-B.

Random selection (the control) and ANOVA F, both at top 100 genes, each
followed by logistic regression under stratified 5-fold CV with PAM50 as a
five-class target. Prints accuracy and macro-F1 per method.

Feature selection sits inside the pipeline, so it is fit on the training
split of each fold only (SCOPE.md 3a: selection inside the fold).

Usage: python scripts/run_baseline.py [config_path]
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from featsel import DataLoader, FeatureSelector  # noqa: E402

CONFIG = "configs/scanb_small.yaml"
METHODS = ["random", "anova_f"]
N_FEATURES = 100
N_SPLITS = 5
SEED = 42


def main(config_path: str = CONFIG) -> None:
    loader = DataLoader(config_path)
    X, y = loader.X, loader.y

    labelled = y.notna()
    if not labelled.all():
        print(f"Dropping {(~labelled).sum()} samples with no PAM50 label")
        X, y = X.loc[labelled], y.loc[labelled]

    print(f"\nClass distribution:\n{y.value_counts().to_string()}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for method in METHODS:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", FeatureSelector(
                method=method, n_features=N_FEATURES, random_state=SEED
            )),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ])

        scores = cross_validate(
            pipe, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1
        )
        acc, f1 = scores["test_accuracy"], scores["test_f1_macro"]
        fit_time = scores["fit_time"]

        print(f"\n=== {N_SPLITS}-fold stratified CV, {method} top-{N_FEATURES} ===")
        print(f"Accuracy : {acc.mean():.4f} +/- {acc.std():.4f}  {np.round(acc, 4)}")
        print(f"Macro-F1 : {f1.mean():.4f} +/- {f1.std():.4f}  {np.round(f1, 4)}")
        print(f"Fit time : {fit_time.mean():.1f}s per fold")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CONFIG)
