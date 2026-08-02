"""Does gaussianizing each gene column improve classification?

Runs the same grid twice, once with standard scaling and once with a rank-based
normal quantile transform, and reports the paired difference in macro-F1.

Usage: python scripts/compare_preprocessing.py [config_path]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from featsel import DataLoader, run_grid, summarize  # noqa: E402

CONFIG = "configs/scanb_small.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
K_VALUES = (50, 250)
SEED = 42
N_SPLITS = 5


def main(config_path: str = CONFIG) -> None:
    loader = DataLoader(config_path)
    X, y = loader.X, loader.y
    labelled = y.notna()
    X, y = X.loc[labelled], y.loc[labelled]

    frames = []
    for preprocess in ("standard", "quantile_normal"):
        print(f"\n=== {preprocess} ===", flush=True)
        scores, _ = run_grid(X, y, k_values=K_VALUES, n_splits=N_SPLITS,
                             seed=SEED, preprocess=preprocess)
        scores["preprocess"] = preprocess
        frames.append(scores)

    raw = pd.concat(frames, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    raw.to_csv(RESULTS_DIR / "preprocessing_raw.csv", index=False)

    summary = pd.concat([
        summarize(f).assign(preprocess=f.preprocess.iloc[0]) for f in frames
    ], ignore_index=True)
    summary.to_csv(RESULTS_DIR / "preprocessing.csv", index=False)

    wide = summary.pivot_table(index=["classifier", "selector", "k"],
                               columns="preprocess", values="macro_f1_mean")
    wide["delta"] = wide["quantile_normal"] - wide["standard"]

    # Paired over folds, so the split is held constant and only the transform varies.
    paired = raw.pivot_table(
        index=["classifier", "selector", "k", "fold"],
        columns="preprocess", values="macro_f1"
    )
    paired["delta"] = paired["quantile_normal"] - paired["standard"]
    by_cell = paired.groupby(["classifier", "selector", "k"])["delta"]
    wide["folds_improved"] = by_cell.apply(lambda d: (d > 0).sum())
    wide["n_folds"] = by_cell.size()

    print("\nMacro-F1: quantile_normal vs standard")
    print(wide.round(4).to_string())
    print(f"\nMean delta over all cells: {wide['delta'].mean():+.4f}")
    print(f"Cells improved: {(wide['delta'] > 0).sum()} of {len(wide)}")
    print("\nMean delta by classifier:")
    print(wide.groupby("classifier")["delta"].mean().round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CONFIG)
