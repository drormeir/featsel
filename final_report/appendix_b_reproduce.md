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
the images (31 MB). It then computes the embeddings and finds the copied
images (about 40 seconds), once each. The embeddings use the GPU when one
is present, else the CPU.

Versions used: Python 3.13, scikit-learn 1.9, NumPy 2.5, SciPy 1.18,
pandas 3.0, PyTorch with CUDA 13. Each run writes a JSON file beside its
CSV with the exact versions, the machine and the git commit.

## The experiments

Chapters 7 and 8 come from the current configs, run on 2026-11-14 and 15.
Each result's JSON file records the commit, machine and settings.

| Result | Command | Output |
|---|---|---|
| Parallel speed-up by worker count (section 6.1) | `.venv/bin/python scripts/parallel_scaling.py --workers 1 2 3 4 5 6 --repeats 3`, then `.venv/bin/python scripts/plot_parallel_scaling.py results/local/parallel_scaling.csv --out final_report/figures/parallel_scaling.png`. Per-job times: `.venv/bin/python scripts/parallel_job_times.py --n-jobs 1 --out <file>`, and the same with `--n-jobs 3` | `results/local/parallel_scaling.csv` and its JSON, copied to [`figures/`](figures/) with the plot and the per-job CSVs |
| Runner against scikit-learn (chapter 6) | `.venv/bin/python scripts/benchmark_speed.py --n-genes 3000 --out <file>` | the table of section 6.1, and `results/speed_benchmark.csv` (see the note below) |
| Speed of each method (chapter 6) | `.venv/bin/python scripts/benchmark_speed_selectors.py <ClassName>`, with `--scanb` for the SCAN-B tasks | `results/local/speed/<ClassName>.csv`, such as `ANOVAFSelector.csv`, and `LassoSelector_scanb.csv` for Lasso on SCAN-B |
| SCAN-B runs (chapter 7) | `.venv/bin/python -m featsel.run --config configs/<name>.yaml --n-jobs 1` for `frontier_scanb70_gpu`, `_berkjones`, `_chains` and `_seq`; `--n-jobs 2` for `frontier_scanb70_cpu`, `_rfecv_auc` and `_all_genes` | `results/local/<name>.csv` |
| SCAN-B tables (chapter 7) | `.venv/bin/python scripts/summarize_frontier.py --n-features 28541 --out final_report/figures/summary_scanb70 results/local/frontier_scanb70_{gpu,cpu,berkjones,rfecv_auc,chains,seq,all_genes}.csv` | `summary_scanb70_chosen.csv`, `_curve.csv`, `_bar.csv` and `_tests.csv` in [`figures/`](figures/) |
| SCAN-B plots (section 7.1) | `.venv/bin/python scripts/plot_frontier.py <the seven CSVs above> --out final_report/figures/frontier_scanb70_roc_auc.png` | one PNG and CSV for all tasks, and one per task |
| BreastMNIST runs (chapter 8) | `.venv/bin/python -m featsel.run --config configs/<name>.yaml --n-jobs 1` for `frontier_breastmnist70_gpu`, `_berkjones`, `_chains`, `_seq` and `_nocopies`; `--n-jobs 2` for `frontier_breastmnist70_cpu` and `_rfecv_auc` | `results/local/<name>.csv` |
| BreastMNIST tables (chapter 8) | `scripts/summarize_frontier.py --n-features 2048` on the six CSVs other than `_nocopies`, with `--out final_report/figures/summary_breastmnist70`; on `_nocopies` alone with `--out final_report/figures/summary_breastmnist70_nocopies`. Paired tests between methods: `.venv/bin/python scripts/paired_tests.py results/local/frontier_breastmnist70_gpu.csv results/local/frontier_breastmnist70_cpu.csv --metric roc_auc --out final_report/figures/paired_tests_breastmnist70.csv` | the `summary_breastmnist70*` and `paired_tests_breastmnist70` CSVs in [`figures/`](figures/) |
| BreastMNIST without copies, plot (section 8.8) | `.venv/bin/python scripts/plot_frontier.py results/local/frontier_breastmnist70_nocopies.csv --out final_report/figures/frontier_breastmnist70_nocopies_roc_auc.png`; it also writes a `_binary` copy, the same plot for the one task, which is deleted | `frontier_breastmnist70_nocopies_roc_auc.png` and `.csv` in [`figures/`](figures/) |
| FGVC-Aircraft (section 8.9) | `.venv/bin/python -m featsel.run --config configs/aircraft70_gpu.yaml --n-jobs 1`; the first call downloads the 2.75 GB archive and computes the embeddings. Tables: `summarize_frontier.py --n-features 2048 --out final_report/figures/summary_aircraft70 results/aircraft70_gpu.csv`, and `paired_tests.py results/aircraft70_gpu.csv --metric roc_auc --out final_report/figures/paired_tests_aircraft70.csv` | `results/aircraft70_gpu.csv`; analysis in `featsel_datasets/fgvc_aircraft/demos/research.ipynb` |
| RFECV, one split, three inner scores (section 7.8) | `.venv/bin/python scripts/rfecv_scoring.py` | prints its table, writes no file |
| PAM50 gene recovery (section 7.6) | `.venv/bin/python scripts/pam50_recovery.py --n-splits 10 --out results/pam50_recovery.md` | [`results/pam50_recovery.md`](../results/pam50_recovery.md), the tables of section 7.6 (the script overwrites its output file) |
| Classifier study (section 7.7) | `.venv/bin/python -m featsel.run --config configs/experiment_classifiers.yaml --n-jobs 4 --max-worker-memory-gb 3` | `results/experiment_classifiers.csv` |

Add `--resume` to any runner command, or to
`scripts/parallel_scaling.py`, to continue a stopped run. For a long
unattended run, start the runner through `scripts/run_until_done.py` with
the same arguments:

```bash
.venv/bin/python scripts/run_until_done.py --config configs/frontier_scanb70_cpu.yaml --n-jobs 2
```

It restarts the runner after a crash and keeps the machine awake.
Section 5.5 describes how runs are protected, and the limits. Git does
not track `results/local/`. The frontier plots and tables used in this
report are copied to [`figures/`](figures/).

**Note on section 6.1.** Its table was measured when the runner fitted
each selection once. The runner now fits each selection 10 times by
default (`selection_repeats`), and `benchmark_speed.py` does not set it
to 1. So a rerun today would not match the table's runner times.

## Determinism

One seed, 42; split `s` uses 42 + `s`. Selections are identical across
runs and devices. Times are not: they depend on the machine and its load.
The frontier records whether repeated fits agreed (`selection_stable`)
and where each ran (`selection_device`).

## Tests

```bash
.venv/bin/python -m pytest -v
```

GPU tests are skipped on a machine without a CUDA GPU.
