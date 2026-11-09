# Appendix C. How to reproduce

[Back to contents](README.md)

Every command runs from the repository root, inside its virtual
environment.

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[gpu,embeddings]"   # PyTorch: GPU paths and the BreastMNIST embeddings
```

The first SCAN-B load downloads GSE96058 (592 MB) into
`featsel_datasets/scanb_gse96058/`. The first BreastMNIST load downloads
the images (31 MB) and computes the embeddings once on the GPU.

Versions used: Python 3.13, scikit-learn 1.9, NumPy 2.5, SciPy 1.18,
pandas 3.0, PyTorch with CUDA 13. Each run writes a JSON file beside its
CSV with the exact versions, the machine and the git commit.

## The experiments

| Result | Command | Output |
|---|---|---|
| Comparison, SCAN-B (chapter 7) | `.venv/bin/python -m featsel.run --config configs/comparison_scanb.yaml --n-jobs 4` | `results/comparison_scanb.csv` |
| Comparison, BreastMNIST (chapter 7) | `.venv/bin/python -m featsel.run --config configs/comparison_breastmnist.yaml --n-jobs -1` | `results/comparison_breastmnist.csv` |
| Its tables | `.venv/bin/python scripts/summarize_comparison.py results/comparison_scanb.csv --n-features 28552 --out <file>` | Markdown tables |
| Paired tests (chapter 7) | `.venv/bin/python scripts/paired_tests.py results/comparison_scanb.csv --metric macro_f1` | table of tests |
| Classifier study (chapter 7) | `.venv/bin/python -m featsel.run --config configs/experiment_classifiers.yaml --n-splits 3` | `results/experiment_classifiers.csv`; figures from `notebooks/02_classifier_behaviour.ipynb` |
| PAM50 gene recovery (chapter 7) | `.venv/bin/python scripts/pam50_recovery.py --n-splits 10 --out <file>` | the tables in `docs/pam50_recovery.md` (the script overwrites its output file, so write elsewhere and paste) |
| Runner against scikit-learn (chapter 6) | `.venv/bin/python scripts/benchmark_speed.py --n-genes 3000 --out <file>` | `docs/speed_benchmark.md` table |
| Speed of each method (chapter 6) | run `featsel/selectors/<name>/demos/speed.ipynb` | `results/local/speed/<name>.csv` |
| Frontier, SCAN-B (chapter 8) | `.venv/bin/python -m featsel.run --config configs/frontier_scanb70_gpu.yaml --n-jobs 1`, then `..._seq.yaml --n-jobs 1`, then `..._cpu.yaml --n-jobs 2` | `results/local/frontier_scanb70_*.csv` |
| Frontier, BreastMNIST (chapter 8) | the same with `configs/frontier_breastmnist70_*.yaml` | `results/local/frontier_breastmnist70_*.csv` |
| RFECV scored by AUC (section 8.6) | `.venv/bin/python -m featsel.run --config configs/frontier_scanb70_rfecv_auc.yaml --n-jobs 2`, and the same with `frontier_breastmnist70_rfecv_auc.yaml`; one split, three inner scores: `.venv/bin/python scripts/rfecv_scoring.py` | `results/local/frontier_*70_rfecv_auc.csv` |
| BreastMNIST at larger `k` (section 8.7) | `.venv/bin/python -m featsel.run --config configs/frontier_breastmnist70_large_k.yaml --n-jobs 1` | `results/local/frontier_breastmnist70_large_k.csv` |
| SCAN-B on all genes (section 8.7) | `.venv/bin/python -m featsel.run --config configs/frontier_scanb70_all_genes.yaml --n-jobs 2` | `results/local/frontier_scanb70_all_genes.csv` |
| Frontier plots | `.venv/bin/python scripts/plot_frontier.py <the three CSVs> --out results/local/frontier_scanb70_roc_auc.png` | one PNG and CSV for all tasks, and one per task |

Add `--resume` to any runner command to continue a stopped run. Files in
`results/local/` are not tracked by git; the frontier plots and tables used
in this report are copied to [`figures/`](figures/).

## Determinism

One seed, 42; split `s` uses 42 + `s`. Selections are identical across
runs and devices. Times are not: they depend on the machine and its load.
The frontier records whether repeated fits agreed (`selection_stable`) and
where each ran (`selection_device`).

## Tests

```bash
.venv/bin/python -m pytest -v
```

GPU tests are skipped on a machine without a CUDA GPU.
