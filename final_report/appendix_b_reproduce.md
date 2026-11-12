# Appendix B. How to reproduce

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
the images (31 MB), computes the embeddings once, and finds the copied
images once (about 40 seconds). The embeddings use the GPU when one is
present, else the CPU.

Versions used: Python 3.13, scikit-learn 1.9, NumPy 2.5, SciPy 1.18,
pandas 3.0, PyTorch with CUDA 13. Each run writes a JSON file beside its
CSV. It records the exact versions, the machine and the git commit.

## The experiments

| Result | Command | Output |
|---|---|---|
| Comparison, SCAN-B (chapter 7) | `.venv/bin/python -m featsel.run --config configs/comparison_scanb.yaml --n-jobs 4 --max-worker-memory-gb 3` | `results/comparison_scanb.csv` |
| Comparison, BreastMNIST (chapter 7) | `.venv/bin/python -m featsel.run --config configs/comparison_breastmnist.yaml --n-jobs -1` | `results/comparison_breastmnist.csv` |
| Berk-Jones on the comparison's splits (chapter 7) | `.venv/bin/python -m featsel.run --config configs/comparison_scanb_berkjones.yaml --n-jobs 4 --max-worker-memory-gb 3`, and the same with `comparison_breastmnist_berkjones.yaml` | `results/comparison_*_berkjones.csv` |
| BreastMNIST without its copies (chapter 7) | `.venv/bin/python -m featsel.run --config configs/comparison_breastmnist_nocopies.yaml --n-jobs 4`. The first BreastMNIST load finds the copy groups and caches them; this needs OpenCV (the `embeddings` extra). The analysis is `featsel_datasets/medmnist_breastmnist/demos/copies.ipynb` | `results/comparison_breastmnist_nocopies.csv` |
| All features on the comparison's splits (section 8.7) | `.venv/bin/python -m featsel.run --config configs/comparison_breastmnist_all_features.yaml --n-jobs 4`, then `.venv/bin/python scripts/paired_tests.py results/comparison_breastmnist.csv results/comparison_breastmnist_all_features.csv results/comparison_breastmnist_berkjones.csv --baseline AllFeatures --out final_report/figures/paired_tests_breastmnist_vs_all_features.csv`. SCAN-B: `comparison_scanb_all_genes.yaml` | `results/comparison_breastmnist_all_features.csv` |
| Parallel speed-up by worker count (section 6.1) | `.venv/bin/python scripts/parallel_scaling.py --workers 1 2 3 4 5 6 --repeats 3`, then `.venv/bin/python scripts/plot_parallel_scaling.py results/local/parallel_scaling.csv --out final_report/figures/parallel_scaling.png`. Per-job times: `.venv/bin/python scripts/parallel_job_times.py --n-jobs 1 --out <file>`, and the same with `--n-jobs 3` | `results/local/parallel_scaling.csv` and its JSON, copied to [`figures/`](figures/) with the plot and the per-job CSVs |
| Comparison tables | `.venv/bin/python scripts/summarize_comparison.py results/comparison_scanb.csv --n-features 28552 --out <file>` | the tables of chapter 7 and Appendix D, in Markdown |
| Paired tests (chapter 7) | `.venv/bin/python scripts/paired_tests.py results/comparison_scanb.csv --metric macro_f1 --out final_report/figures/paired_tests_comparison_scanb.csv`, and the same for BreastMNIST | the tables of tests in [`figures/`](figures/) |
| Classifier study (chapter 7) | `.venv/bin/python -m featsel.run --config configs/experiment_classifiers.yaml --n-splits 3` | `results/experiment_classifiers.csv`; figures from `notebooks/02_classifier_behaviour.ipynb` |
| PAM50 gene recovery (chapter 7) | `.venv/bin/python scripts/pam50_recovery.py --n-splits 10 --out <file>` | the tables of section 7.6, in Markdown (the script overwrites its output file) |
| Runner against scikit-learn (chapter 6) | `.venv/bin/python scripts/benchmark_speed.py --n-genes 3000 --out <file>` | the table of section 6.1, and `results/speed_benchmark.csv` (see the note below) |
| Speed of each method (chapter 6) | run `featsel/selectors/<name>/demos/speed.ipynb` | `results/local/speed/<ClassName>.csv`, such as `ANOVAFSelector.csv`, and `LassoSelector_scanb.csv` for Lasso on SCAN-B |
| Frontier, SCAN-B (chapter 8) | `.venv/bin/python -m featsel.run --config configs/frontier_scanb70_gpu.yaml --n-jobs 1`, then `..._seq.yaml --n-jobs 1`, then `..._cpu.yaml --n-jobs 2` | `results/local/frontier_scanb70_*.csv` |
| Frontier, SCAN-B, 16 splits (section 8.3) | `.venv/bin/python scripts/run_until_done.py --config configs/frontier_scanb70_gpu.yaml --n-jobs 1 --n-splits 16`, the same with `frontier_scanb70_berkjones.yaml`, and with `frontier_scanb70_tree.yaml --n-jobs 2` (its CSV starts as a copy of the cpu run's tree importance rows). Then `.venv/bin/python scripts/paired_tests.py results/local/frontier_scanb70_gpu.csv results/local/frontier_scanb70_berkjones.csv results/local/frontier_scanb70_tree.csv --metric roc_auc --own-count --out final_report/figures/paired_tests_frontier_scanb70_16_splits.csv`, and `.venv/bin/python scripts/plot_paired_differences.py final_report/figures/paired_tests_frontier_scanb70_16_splits.csv --reference MannWhitneySelector --out final_report/figures/frontier_scanb70_16_splits.png` | the three CSVs, extended from 5 splits to 16; the tests and the figure in [`figures/`](figures/) |
| Frontier, BreastMNIST (chapter 8) | the same with `configs/frontier_breastmnist70_*.yaml` | `results/local/frontier_breastmnist70_*.csv` |
| RFECV scored by AUC (section 8.6) | `.venv/bin/python -m featsel.run --config configs/frontier_scanb70_rfecv_auc.yaml --n-jobs 2`, and the same with `frontier_breastmnist70_rfecv_auc.yaml`. One split, three inner scores: `.venv/bin/python scripts/rfecv_scoring.py` | `results/local/frontier_*70_rfecv_auc.csv`. The script prints its table and writes no file. |
| BreastMNIST at larger `k` (section 8.7) | `.venv/bin/python -m featsel.run --config configs/frontier_breastmnist70_large_k.yaml --n-jobs 1` | `results/local/frontier_breastmnist70_large_k.csv` |
| SCAN-B on all genes (section 8.7) | `.venv/bin/python -m featsel.run --config configs/frontier_scanb70_all_genes.yaml --n-jobs 2` | `results/local/frontier_scanb70_all_genes.csv` |
| Berk-Jones, chains and BreastMNIST without copies on the frontier (chapter 8) | `.venv/bin/python -m featsel.run --config configs/<name>.yaml --n-jobs 1`, for `frontier_scanb70_berkjones`, `frontier_breastmnist70_berkjones`, `frontier_scanb70_chains`, `frontier_breastmnist70_chains` and `frontier_breastmnist70_nocopies` | `results/local/<name>.csv` |
| FGVC-Aircraft (section 8.8) | `.venv/bin/python -m featsel.run --config configs/aircraft70_gpu.yaml --n-jobs 1`; the first call downloads the 2.75 GB archive and computes the embeddings | `results/aircraft70_gpu.csv`; analysis in `featsel_datasets/fgvc_aircraft/demos/research.ipynb` |
| Frontier plots | `.venv/bin/python scripts/plot_frontier.py <the three CSVs> --out results/local/frontier_scanb70_roc_auc.png` | one PNG and CSV for all tasks, and one per task |

Add `--resume` to any runner command to continue a stopped run. For a
long unattended run, start it through `scripts/run_until_done.py` with the
same arguments, for example `.venv/bin/python scripts/run_until_done.py
--config configs/comparison_scanb_all_genes.yaml --n-jobs 4
--max-worker-memory-gb 3`. It restarts the runner after a crash and keeps
the machine awake. `scripts/parallel_scaling.py` takes `--resume` too. How
the runs are protected, and the limits: section 5.5. Files in
`results/local/` are not tracked by git. The frontier plots and tables used
in this report are copied to [`figures/`](figures/).

The table of section 6.1 was measured when the runner fitted each
selection once. It now fits each selection 10 times by default
(`selection_repeats`).
`benchmark_speed.py` does not set it to 1. So a rerun today times 10 fits
per selection on the runner's side. Its runner times would not match the
table.

## Determinism

One seed, 42. Split `s` uses 42 + `s`. Selections are identical across
runs and devices. Times are not. They depend on the machine and its load.
The frontier records whether repeated fits agreed (`selection_stable`) and
where each ran (`selection_device`).

## Tests

```bash
.venv/bin/python -m pytest -v
```

GPU tests are skipped on a machine without a CUDA GPU.
