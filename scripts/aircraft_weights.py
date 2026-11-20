"""
FGVC-Aircraft: ResNet-50's ImageNet weights V1 against V2, with all features.

For each weights version, on the 1,200 photos of the task (Boeing 737
against the Airbus A320 family): the share of exact zeros in their
embeddings, and the test AUC of logistic regression on every non-constant
feature. The protocol is that of configs/aircraft70_gpu.yaml: 16 splits,
70% of the photos in the training set, seed 42. The runner's own CSV for
each version goes to results/local/.

The first call for a version embeds the whole archive (on the GPU when
there is one) and caches the result beside the loader.

    .venv/bin/python scripts/aircraft_weights.py --out results/aircraft_weights.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from featsel import environment
from featsel.run import run
from featsel_datasets.fgvc_aircraft.aircraft import FGVCAircraft
from featsel_datasets.fgvc_aircraft.featsel_view import NAME, TARGET, load_xy

WEIGHTS = ('V1', 'V2')
CONFIG = {
    'seed': 42,
    'n_splits': 16,
    'train_sizes': [0.7],
    'preprocess': ['standard'],
    'selection_repeats': 1,
    'metrics': ['roc_auc'],
    'models': [{'name': 'logistic_regression', 'params': {'max_iter': 1000}}],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--out', type=Path, required=True, help='Output CSV')
    args = parser.parse_args()

    frames = []
    for weights in WEIGHTS:
        data = FGVCAircraft(weights=weights)
        X, y = load_xy(data)
        zero_share = (data.embeddings.loc[y.index].to_numpy() == 0).mean()
        # Random selection of every feature is the all-features reference.
        config = {**CONFIG,
                  'selectors': [{'name': 'RandomSelector', 'k': [X.shape[1]]}],
                  'output': f'results/local/aircraft_weights_{weights.lower()}.csv'}
        rows = pd.read_csv(run(X, y, config, target=TARGET, name=f'{NAME} {weights}'))
        frames.append(pd.DataFrame({'weights': weights, 'zero_share': zero_share,
                                    'n_features': X.shape[1], 'split': rows['split'],
                                    'roc_auc': rows['roc_auc']}))

    table = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    environment.write_json(args.out.with_suffix('.json'), config=CONFIG, weights=WEIGHTS)
    print(table.groupby('weights')[['zero_share', 'n_features', 'roc_auc']].mean())


if __name__ == '__main__':
    main()
