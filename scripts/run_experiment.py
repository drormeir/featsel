"""Run the feature-selection comparison grid and write the results to disk.

Usage: python scripts/run_experiment.py [config_path]

Writes three tables under results/:
  scores_raw.csv  - one row per (selector, k, classifier, fold)
  scores.csv      - mean and std over folds
  stability.csv   - Kuncheva consistency index per (selector, k)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from featsel import DataLoader, kuncheva_index, run_grid, summarize  # noqa: E402
from featsel.experiment import CLASSIFIERS, K_VALUES, SELECTORS  # noqa: E402

CONFIG = "configs/scanb_small.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42
N_SPLITS = 5


def main(config_path: str = CONFIG) -> None:
    loader = DataLoader(config_path)
    X, y = loader.X, loader.y

    labelled = y.notna()
    if not labelled.all():
        print(f"Dropping {(~labelled).sum()} samples with no {loader.target_column} label")
        X, y = X.loc[labelled], y.loc[labelled]

    n_cells = len(SELECTORS) * len(K_VALUES) * len(CLASSIFIERS) * N_SPLITS
    print(f"Grid: {len(SELECTORS)} selectors x {len(K_VALUES)} k x "
          f"{len(CLASSIFIERS)} classifiers x {N_SPLITS} folds = {n_cells} fits")

    scores, supports = run_grid(X, y, n_splits=N_SPLITS, seed=SEED)
    summary = summarize(scores)
    stability = kuncheva_index(supports, n_total_features=X.shape[1])

    RESULTS_DIR.mkdir(exist_ok=True)
    scores.to_csv(RESULTS_DIR / "scores_raw.csv", index=False)
    summary.to_csv(RESULTS_DIR / "scores.csv", index=False)
    stability.to_csv(RESULTS_DIR / "stability.csv", index=False)

    print(f"\nWrote {len(scores)} rows to {RESULTS_DIR}/")
    print("\nMacro-F1 by selector, k and classifier:")
    pivot = summary.pivot_table(
        index=["classifier", "k"], columns="selector", values="macro_f1_mean"
    )
    print(pivot.round(4).to_string())

    print("\nSelection stability (Kuncheva consistency index):")
    print(stability.pivot_table(index="k", columns="selector",
                                values="consistency_index").round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CONFIG)
