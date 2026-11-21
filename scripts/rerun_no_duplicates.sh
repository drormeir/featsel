#!/usr/bin/env bash
# Rerun every image-data study without duplicates (docs/DECISIONS.md,
# 2026-11-21), one run after another, then rebuild its tables.
#
#   BreastMNIST: the main runs without copies already exist
#   (frontier_breastmnist70_nocopies, chain_size_breastmnist70_nocopies);
#   this adds Sequential selection, RFE's C, the classifier's C and the tree
#   count.
#   FGVC-Aircraft: every run again, without two near-identical photos
#   (featsel_datasets/fgvc_aircraft/demos/duplicates.ipynb).
#
# About 65 minutes on the report's laptop. Run from anywhere:
#
#   bash scripts/rerun_no_duplicates.sh
#
# Stops at the first error; each step prints its start time. The tables in
# detailed_report/figures/ and results/ are overwritten; the report text is
# updated afterwards (detailed report, section 10.1).

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
F=detailed_report/figures
L=results/local

step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }

# --- runs ---------------------------------------------------------------
step "BreastMNIST without copies: Sequential (about 2 min)"
$PY -m featsel.run --config configs/frontier_breastmnist70_nocopies_seq.yaml --n-jobs 1
step "BreastMNIST without copies: RFE's C (about 1 min)"
$PY -m featsel.run --config configs/rfe_c_breastmnist70.yaml --n-jobs 2
step "BreastMNIST without copies: the classifier's C (about 9 min)"
$PY -m featsel.run --config configs/clf_c_breastmnist70.yaml --n-jobs 1

step "FGVC-Aircraft: main run (about 9 min)"
$PY -m featsel.run --config configs/aircraft70_gpu.yaml --n-jobs 1
step "FGVC-Aircraft: chain size (about 8 min)"
$PY -m featsel.run --config configs/chain_size_aircraft70.yaml --n-jobs 1
step "FGVC-Aircraft: RFE's C (about 7 min)"
$PY -m featsel.run --config configs/rfe_c_aircraft70.yaml --n-jobs 1
step "FGVC-Aircraft: the classifier's C (about 22 min)"
$PY -m featsel.run --config configs/clf_c_aircraft70.yaml --n-jobs 1
step "FGVC-Aircraft: weights V1 against V2, by test and by validation AUC (about 2 min)"
$PY scripts/aircraft_weights.py --out results/aircraft_weights.csv
$PY scripts/aircraft_weights_validation.py --out results/aircraft_weights_validation.csv

step "Tree count, image datasets (minutes; SCAN-B rows kept)"
$PY scripts/tree_count_stability.py --datasets BreastMNIST FGVC-Aircraft \
    --out results/tree_count_stability.csv

# --- tables (read the saved CSVs, run nothing) ----------------------------
step "Tables: BreastMNIST without copies"
$PY scripts/summarize_frontier.py --n-features 2048 --out $F/summary_breastmnist70_nocopies \
    $L/frontier_breastmnist70_nocopies.csv $L/frontier_breastmnist70_nocopies_seq.csv
$PY scripts/summarize_side_metrics.py --n-features 2048 --out $F/summary_breastmnist70_nocopies_side \
    $L/frontier_breastmnist70_nocopies.csv $L/frontier_breastmnist70_nocopies_seq.csv
$PY scripts/paired_tests.py $L/frontier_breastmnist70_nocopies.csv --metric roc_auc \
    --out $F/paired_tests_breastmnist70_nocopies.csv
$PY scripts/plot_frontier.py $L/frontier_breastmnist70_nocopies.csv \
    $L/frontier_breastmnist70_nocopies_seq.csv --out $F/frontier_breastmnist70_nocopies_roc_auc.png
rm -f $F/frontier_breastmnist70_nocopies_roc_auc_binary.png $F/frontier_breastmnist70_nocopies_roc_auc_binary.csv
$PY scripts/rfe_c_tests.py $L/frontier_breastmnist70_nocopies.csv $L/rfe_c_breastmnist70.csv \
    --out $F/rfe_c_breastmnist70.csv
$PY scripts/clf_c_tests.py $L/clf_c_breastmnist70.csv --out $F/clf_c_breastmnist70.csv

step "Tables: FGVC-Aircraft"
$PY scripts/summarize_frontier.py --n-features 2048 --out $F/summary_aircraft70 results/aircraft70_gpu.csv
$PY scripts/summarize_side_metrics.py --n-features 2048 --out $F/summary_aircraft70_side results/aircraft70_gpu.csv
$PY scripts/paired_tests.py results/aircraft70_gpu.csv --metric roc_auc --out $F/paired_tests_aircraft70.csv
$PY scripts/plot_frontier.py results/aircraft70_gpu.csv --out $F/frontier_aircraft70_roc_auc.png
rm -f $F/frontier_aircraft70_roc_auc_binary.png $F/frontier_aircraft70_roc_auc_binary.csv
$PY scripts/chain_size_tests.py results/aircraft70_gpu.csv $L/chain_size_aircraft70.csv \
    --out $F/chain_size_aircraft70.csv
$PY scripts/rfe_c_tests.py results/aircraft70_gpu.csv $L/rfe_c_aircraft70.csv --out $F/rfe_c_aircraft70.csv
$PY scripts/clf_c_tests.py $L/clf_c_aircraft70.csv --out $F/clf_c_aircraft70.csv

step "Tables: RFECV by AUC against by accuracy"
$PY scripts/rfecv_scoring_tests.py \
    scanb=$L/frontier_scanb70_cpu.csv,$L/frontier_scanb70_rfecv_auc.csv \
    breastmnist=$L/frontier_breastmnist70_nocopies.csv \
    aircraft=results/aircraft70_gpu.csv \
    --out $F/rfecv_scoring_tests.csv

# --- notebooks that read the aircraft results ------------------------------
step "Notebooks: FGVC-Aircraft weights and research"
.venv/bin/jupyter nbconvert --to notebook --execute --inplace featsel_datasets/fgvc_aircraft/demos/weights.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace featsel_datasets/fgvc_aircraft/demos/research.ipynb

step "Done. Next: update the report text (detailed report, section 10.1)."
