# Appendix A. How to reproduce

[Back to contents](README.md)

Every command runs from the repository root, inside its virtual
environment.

## A.1 Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[gpu,embeddings]"   # PyTorch: GPU paths and the BreastMNIST embeddings
```

The first SCAN-B load downloads GSE96058 (592 MB) into
`featsel_datasets/scanb_gse96058/`. The first BreastMNIST load downloads
the images (31 MB). It then computes the embeddings, and finds the copied
images in about 40 seconds. Each step runs once, and its result is
cached. The embeddings use the GPU when one is present, else the CPU.

Versions used: Python 3.13, scikit-learn 1.9, NumPy 2.5, SciPy 1.18,
pandas 3.0, PyTorch with CUDA 13. Each run writes a JSON file beside its
CSV with the exact versions, the machine and the git commit.

## A.2 The experiments

Chapters 7 and 8 come from the current configs. The main SCAN-B runs
were made on 2026-11-14 and 15; the later runs, from 2026-11-20 on,
among them BreastMNIST without copies, carry their dates in their JSON files. Each result's JSON
file records the commit, machine and settings.

| Result | Command | Output |
|---|---|---|
| Parallel speed-up by worker count (section 6.1) | `.venv/bin/python scripts/parallel_scaling.py --workers 1 2 3 4 5 6 --repeats 3`, then `.venv/bin/python scripts/plot_parallel_scaling.py results/local/parallel_scaling.csv --out detailed_report/figures/parallel_scaling.png`. Per-job times: `.venv/bin/python scripts/parallel_job_times.py --n-jobs 1 --out <file>`, and the same with `--n-jobs 3` | `results/local/parallel_scaling.csv` and its JSON, copied to [`figures/`](figures/) with the plot and the per-job CSVs |
| Runner against scikit-learn (section 6.1) | `.venv/bin/python scripts/benchmark_speed.py --n-genes 3000 --out <file>` | the table of section 6.1, and `results/speed_benchmark.csv` (see the note below) |
| Tree importance on the runs' training set, alone and two at once (sections 6.2, 7.11) | `.venv/bin/python scripts/tree_timing.py --out results/tree_timing.csv`; run it on an idle machine | [`results/tree_timing.csv`](../results/tree_timing.csv) and its JSON |
| Speed of each method (section 6.2, Appendix D) | `.venv/bin/python scripts/benchmark_speed_selectors.py <ClassName>`, with `--scanb` for the SCAN-B tasks | `results/local/speed/<ClassName>.csv`, such as `ANOVAFSelector.csv`, and `LassoSelector_scanb.csv` for Lasso on SCAN-B. The RFE, RFECV and Lasso files that the report cites are copied, with their JSON, to [`results/speed/`](../results/speed/) |
| Speed on real SCAN-B (the second table of section 6.2) | `.venv/bin/python scripts/benchmark_selectors.py --n-genes 1000 3000 10000 28552 --repeats 5 --max-seconds 120` | `results/local/benchmark_selectors/<start>.csv` and its JSON. The series the report cites is copied to [`results/speed/scanb_basal_2026-11-07.csv`](../results/speed/scanb_basal_2026-11-07.csv), with its JSON. It (2026-11-07) predates the removal of liblinear: its `liblinear` rows are the "scikit-learn's solver" column, and a rerun today gives scikit-learn at its defaults instead |
| SCAN-B runs (chapter 7) | `.venv/bin/python -m featsel.run --config configs/<name>.yaml --n-jobs 1` for `frontier_scanb70_gpu`, `_berkjones`, `_chains` and `_seq`; `--n-jobs 2` for `frontier_scanb70_cpu`, `_rfecv_auc` and `_all_genes` | `results/local/<name>.csv` |
| SCAN-B tables (chapter 7) | `.venv/bin/python scripts/summarize_frontier.py --n-features 28541 --out detailed_report/figures/summary_scanb70 results/local/frontier_scanb70_{gpu,cpu,berkjones,rfecv_auc,chains,seq,all_genes}.csv` | `summary_scanb70_chosen.csv`, `_curve.csv`, `_bar.csv` and `_tests.csv` in [`figures/`](figures/) |
| Corrected against plain t-test counts (section 9.3, item 9) | `.venv/bin/python scripts/correction_effect.py --n-test 893 --n-train 2083 detailed_report/figures/summary_scanb70` | printed: 290 comparisons, 164 established by the corrected test, 233 by a plain paired t-test; against all genes 20 and 52; reads the saved tables and runs nothing |
| Side scores and memory at the chosen `k`: macro-F1 with quartiles, balanced accuracy, peak selection memory | `.venv/bin/python scripts/summarize_side_metrics.py --n-features 28541 --out detailed_report/figures/summary_scanb70_side` on the seven SCAN-B CSVs above; `.venv/bin/python scripts/summarize_side_metrics.py --n-features 2048 --out detailed_report/figures/summary_breastmnist70_nocopies_side results/local/frontier_breastmnist70_nocopies.csv`, and with `--out detailed_report/figures/summary_aircraft70_side results/aircraft70_gpu.csv` | `summary_scanb70_side.csv`, `summary_breastmnist70_nocopies_side.csv` and `summary_aircraft70_side.csv` (`results/aircraft70_gpu.csv`) in [`figures/`](figures/), for sections 7.12 and 8.10; reads the saved CSVs and runs no experiment |
| AUC against `k` (sections 7.2 and 8.2) | `.venv/bin/python scripts/plot_curves.py --out detailed_report/figures/curves_scanb70.png "SCAN-B=detailed_report/figures/summary_scanb70_curve.csv"`; and `.venv/bin/python scripts/plot_curves.py --out detailed_report/figures/curves_images70.png "BreastMNIST=detailed_report/figures/summary_breastmnist70_nocopies_curve.csv" "FGVC-Aircraft=detailed_report/figures/summary_aircraft70_curve.csv"` | `curves_scanb70.png` and `curves_images70.png` in [`figures/`](figures/); reads the curve tables and runs nothing |
| SCAN-B plots (section 7.1) | `.venv/bin/python scripts/plot_frontier.py <the seven CSVs above> --out detailed_report/figures/frontier_scanb70_roc_auc.png` | one PNG and CSV for all tasks, and one per task |
| BreastMNIST runs (chapter 8), without the copied images | `.venv/bin/python -m featsel.run --config configs/frontier_breastmnist70_nocopies.yaml --n-jobs 1`; Sequential selection: `configs/frontier_breastmnist70_nocopies_seq.yaml`, not yet run (section 10.1). The older configs `frontier_breastmnist70_gpu`, `_cpu`, `_berkjones`, `_chains`, `_seq` and `_rfecv_auc` run the data with copies, for `demos/copies.ipynb` | `results/local/<name>.csv` |
| BreastMNIST tables (chapter 8) | `.venv/bin/python scripts/summarize_frontier.py --n-features 2048 --out detailed_report/figures/summary_breastmnist70_nocopies results/local/frontier_breastmnist70_nocopies.csv`. Paired tests between methods: `.venv/bin/python scripts/paired_tests.py results/local/frontier_breastmnist70_nocopies.csv --metric roc_auc --out detailed_report/figures/paired_tests_breastmnist70_nocopies.csv` | the `summary_breastmnist70_nocopies*` and `paired_tests_breastmnist70_nocopies` CSVs in [`figures/`](figures/) |
| BreastMNIST plot (chapter 8) | `.venv/bin/python scripts/plot_frontier.py results/local/frontier_breastmnist70_nocopies.csv --out detailed_report/figures/frontier_breastmnist70_nocopies_roc_auc.png`; the `_binary` copy is deleted | `frontier_breastmnist70_nocopies_roc_auc.png` and `.csv` in [`figures/`](figures/) |
| FGVC-Aircraft (section 8.9) | `.venv/bin/python -m featsel.run --config configs/aircraft70_gpu.yaml --n-jobs 1`; the first call downloads the 2.75 GB archive and computes the embeddings. Tables: `.venv/bin/python scripts/summarize_frontier.py --n-features 2048 --out detailed_report/figures/summary_aircraft70 results/aircraft70_gpu.csv`, and `.venv/bin/python scripts/paired_tests.py results/aircraft70_gpu.csv --metric roc_auc --out detailed_report/figures/paired_tests_aircraft70.csv`. Plot: `.venv/bin/python scripts/plot_frontier.py results/aircraft70_gpu.csv --out detailed_report/figures/frontier_aircraft70_roc_auc.png`; the `_binary` copy is deleted | `results/aircraft70_gpu.csv`, `frontier_aircraft70_roc_auc.png` and `.csv` in [`figures/`](figures/); analysis in `featsel_datasets/fgvc_aircraft/demos/research.ipynb` |
| BreastMNIST, ResNet-50 weights V1 against V2 (section 2.2) | `.venv/bin/jupyter nbconvert --to notebook --execute --inplace featsel_datasets/medmnist_breastmnist/demos/weights.ipynb`; the first V1 call embeds the 780 images | [`results/breastmnist_weights.csv`](../results/breastmnist_weights.csv) and its JSON |
| FGVC-Aircraft, ResNet-50 weights V1 against V2 (section 2.3) | `.venv/bin/python scripts/aircraft_weights.py --out results/aircraft_weights.csv`; the first V2 call embeds the archive again | [`results/aircraft_weights.csv`](../results/aircraft_weights.csv) and its JSON |
| BreastMNIST with the copies, 409 training images (for `demos/copies.ipynb`; section 8.8) | `.venv/bin/python -m featsel.run --config configs/frontier_breastmnist70_n409.yaml --n-jobs 1`, about 12 minutes. Tables: `.venv/bin/python scripts/summarize_frontier.py --n-features 2048 --out detailed_report/figures/summary_breastmnist70_n409 results/local/frontier_breastmnist70_n409.csv`. All features in the three cases: the last section of `featsel_datasets/medmnist_breastmnist/demos/copies.ipynb` | `results/local/frontier_breastmnist70_n409.csv` and its JSON; the `summary_breastmnist70_n409` CSVs in [`figures/`](figures/); [`results/breastmnist_copies_size.csv`](../results/breastmnist_copies_size.csv) |
| Every image-data study without duplicates (section 10.1) | `bash scripts/rerun_no_duplicates.sh`, about 65 minutes: the runs and tables of the rows above for BreastMNIST without copies and FGVC-Aircraft without two near-identical photos, one after another | the files of those rows, overwritten |
| FGVC-Aircraft, the same choice on validation sets (section 2.3) | `.venv/bin/python scripts/aircraft_weights_validation.py --out results/aircraft_weights_validation.csv`, about 1 minute | [`results/aircraft_weights_validation.csv`](../results/aircraft_weights_validation.csv) and its JSON |
| The classifier's C, 0.01 to 100 against 1 (section 3.8) | `.venv/bin/python -m featsel.run --config configs/<name>.yaml --n-jobs 1` for `clf_c_breastmnist70` and `clf_c_aircraft70`, about 9 and 22 minutes. Tables: `.venv/bin/python scripts/clf_c_tests.py results/local/<name>.csv --out detailed_report/figures/<name>.csv` | `results/local/clf_c_*.csv` and their JSON; `clf_c_breastmnist70.csv` and `clf_c_aircraft70.csv` in [`figures/`](figures/) |
| Tree importance's ranking against the forest's seed (section 7.4) | `.venv/bin/python scripts/tree_count_stability.py --out results/tree_count_stability.csv`, about 50 minutes | [`results/tree_count_stability.csv`](../results/tree_count_stability.csv) and its JSON |
| RFE's C, 0.001 to 1 against 0.01 (section 7.9) | `.venv/bin/python -m featsel.run --config configs/<name>.yaml` for `rfe_c_scanb70` and `rfe_c_breastmnist70` (`--n-jobs 2`) and `rfe_c_aircraft70` (`--n-jobs 1`); about 26, 1 and 7 minutes. Tables: `.venv/bin/python scripts/rfe_c_tests.py <main run> results/local/<name>.csv --out detailed_report/figures/<name>.csv`, with `--every-k` for SCAN-B, where the main run is `results/local/frontier_scanb70_cpu.csv`, `results/local/frontier_breastmnist70_nocopies.csv` and `results/aircraft70_gpu.csv`. The saved BreastMNIST table still comes from the data with copies (section 10.1) | `results/local/rfe_c_*.csv` and their JSON; `rfe_c_scanb70.csv`, `rfe_c_breastmnist70.csv` and `rfe_c_aircraft70.csv` in [`figures/`](figures/) |
| Chain first-step size, 250 and 1,000 against 500 (section 8.7) | `.venv/bin/python -m featsel.run --config configs/<name>.yaml --n-jobs 1` for `chain_size_breastmnist70_nocopies` and `chain_size_aircraft70`, about 8 minutes each. Tables: `.venv/bin/python scripts/chain_size_tests.py <500 run> results/local/<name>.csv --out detailed_report/figures/<name>.csv`, where the 500 run is `results/local/frontier_breastmnist70_nocopies.csv` and `results/aircraft70_gpu.csv` | `results/local/chain_size_*.csv` and their JSON; `chain_size_breastmnist70_nocopies.csv` and `chain_size_aircraft70.csv` in [`figures/`](figures/) |
| Simulated rare signal (section 8.11) | `.venv/bin/python -m featsel.run --config configs/rare_signal70.yaml --n-jobs 1`, about 5 minutes. Tables: `.venv/bin/python scripts/summarize_frontier.py --n-features 10000 --out detailed_report/figures/summary_rare_signal70 results/rare_signal70.csv`, then `.venv/bin/python scripts/rare_signal_tests.py results/rare_signal70.csv --out detailed_report/figures/paired_tests_rare_signal70.csv`; the `_bar` table is deleted | `results/rare_signal70.csv` and its JSON; `summary_rare_signal70_chosen.csv`, `_curve.csv`, `_tests.csv` and `paired_tests_rare_signal70.csv` in [`figures/`](figures/) |
| RFECV, one split, three inner scores (a check behind section 7.8, not reported) | `.venv/bin/python scripts/rfecv_scoring.py` | prints its table, writes no file |
| liblinear against the exact solvers, from the archived run (section 6.2) | `.venv/bin/python scripts/liblinear_vs_exact.py results/local/archive_2026-11-13 --out detailed_report/figures/liblinear_vs_exact.csv`; the archived run cannot be made again | [`figures/liblinear_vs_exact.csv`](figures/liblinear_vs_exact.csv) |
| Every pair of methods at each `k`, SCAN-B (section 7.9) | `.venv/bin/python scripts/paired_tests.py results/local/frontier_scanb70_{gpu,cpu,chains}.csv --metric roc_auc --out detailed_report/figures/paired_tests_scanb70.csv` | [`figures/paired_tests_scanb70.csv`](figures/paired_tests_scanb70.csv) |
| Every pair of methods at their chosen `k` (sections 7.1, 8.1, 8.9) | `.venv/bin/python scripts/chosen_pair_tests.py results/local/frontier_scanb70_{gpu,cpu,chains,berkjones,rfecv_auc}.csv --within 0.01 --out detailed_report/figures/chosen_pairs_scanb70.csv`; the same with `results/local/frontier_breastmnist70_nocopies.csv` and `results/aircraft70_gpu.csv`, out `chosen_pairs_breastmnist70_nocopies.csv` and `chosen_pairs_aircraft70.csv` | `chosen_pairs_*.csv` in [`figures/`](figures/) |
| Mann-Whitney 1000 then RFE against every method at 10 genes (section 7.9) | `.venv/bin/python scripts/k10_tests.py results/local/frontier_scanb70_{gpu,cpu,chains,seq}.csv --out detailed_report/figures/scanb70_k10_tests.csv` | [`figures/scanb70_k10_tests.csv`](figures/scanb70_k10_tests.csv) |
| RFECV by AUC against by accuracy (sections 7.8, 8.6) | `.venv/bin/python scripts/rfecv_scoring_tests.py scanb=results/local/frontier_scanb70_cpu.csv,results/local/frontier_scanb70_rfecv_auc.csv breastmnist=results/local/frontier_breastmnist70_cpu.csv,results/local/frontier_breastmnist70_rfecv_auc.csv breastmnist_nocopies=results/local/frontier_breastmnist70_nocopies.csv aircraft=results/aircraft70_gpu.csv --out detailed_report/figures/rfecv_scoring_tests.csv` | [`figures/rfecv_scoring_tests.csv`](figures/rfecv_scoring_tests.csv) |
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

**Note on section 6.1.** The runner fits each selection 10 times by
default (`selection_repeats`). `benchmark_speed.py` sets it to 1, as
scikit-learn fits once, and turns off the runner's memory cap.

## A.3 Determinism

Seeds are described in section 3.9. Selections are identical across
runs. They are identical across devices too, except Lasso: on LumA and
LumB, PPDNA on the CPU and on the GPU differ by 1 to 3 genes
(section 6.2). Times are not identical: they depend on the machine and
its load.
The frontier records whether repeated fits agreed (`selection_stable`)
and where each ran (`selection_device`).

## A.4 Tests

```bash
.venv/bin/python -m pytest -v
```

GPU tests are skipped on a machine without a CUDA GPU.
