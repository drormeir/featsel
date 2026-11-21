"""
FGVC-Aircraft: ResNet-50's weights V1 against V2, chosen on validation sets.

scripts/aircraft_weights.py compares the two by test AUC, on the 16 splits
that later test the methods. This script makes the same choice without the
test set. On each split of configs/aircraft70_gpu.yaml (16 splits, 70%
training, seed 42), the training set is standardized and cut into 5
stratified folds, as the runner does for k (featsel/run.py,
_validation_auc: seed 42 + split). Logistic regression on all
non-constant features is fit on four folds and scored by ROC AUC on the
fifth. The version with the higher mean validation AUC wins that split.

    .venv/bin/python scripts/aircraft_weights_validation.py \
        --out results/aircraft_weights_validation.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from featsel import environment
from featsel_datasets.fgvc_aircraft.aircraft import FGVCAircraft
from featsel_datasets.fgvc_aircraft.featsel_view import load_xy

WEIGHTS = ('V1', 'V2')
SEED = 42
N_SPLITS = 16
TRAIN_SIZE = 0.7
FOLDS = 5


def validation_auc(X, y, train_idx, seed):
    """Mean ROC AUC over the folds of one standardized training set."""
    X_train = StandardScaler().fit_transform(X[train_idx]).astype(np.float32)
    y_train = y[train_idx]
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    scores = []
    for fit, valid in folds.split(np.zeros((len(y_train), 1)), y_train):
        model = LogisticRegression(max_iter=1000).fit(X_train[fit], y_train[fit])
        scores.append(roc_auc_score(y_train[valid], model.predict_proba(X_train[valid])[:, 1]))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--out', type=Path, required=True, help='Output CSV')
    args = parser.parse_args()

    data = {}
    for weights in WEIGHTS:
        X, y = load_xy(FGVCAircraft(weights=weights), drop_copies=True)
        data[weights] = (np.asarray(X, dtype=np.float32), y)
    labels = data['V1'][1]
    assert all(d[1].index.equals(labels.index) for d in data.values())
    y = pd.factorize(labels, sort=True)[0]

    splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, train_size=TRAIN_SIZE,
                                      random_state=SEED)
    rows = []
    for split, (train_idx, _) in enumerate(splitter.split(data['V1'][0], y)):
        row = {'split': split}
        for weights in WEIGHTS:
            row[f'validation_auc_{weights.lower()}'] = validation_auc(
                data[weights][0], y, train_idx, SEED + split)
        row['chosen'] = max(WEIGHTS, key=lambda w: row[f'validation_auc_{w.lower()}'])
        rows.append(row)
        print(row)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    environment.write_json(args.out.with_suffix('.json'), seed=SEED, n_splits=N_SPLITS,
                           train_size=TRAIN_SIZE, folds=FOLDS, weights=WEIGHTS)
    print(table.drop(columns='split').describe().round(4))
    print(table['chosen'].value_counts())


if __name__ == '__main__':
    main()
