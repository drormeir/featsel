# Configuration Files

An **experiment config** describes one comparison and names its dataset. The
dataset is either a Python module in `featsel_datasets/` (SCAN-B) or a
**dataset config** describing CSV files. Every path in a config is relative
to the file itself.

## Files

| File | Kind | Description |
|------|------|-------------|
| `template.yaml` | dataset | Documented template for a new CSV dataset config |
| `frontier_*70_*.yaml` | experiment | The cost-quality frontier at 70% training: GPU, CPU and Sequential runs on the same splits; `_berkjones`, `_chains` (two-stage selection), `_nocopies`, `_rfecv_auc` and `_all_genes` add to it |
| `experiment_classifiers.yaml` | experiment | Random vs Mann-Whitney across five classifiers, 12 splits |
| `smoke_scanb.yaml` | experiment | Smoke test: every selector and classifier once on one SCAN-B split, about 3 minutes on 6 workers. A check, not a result |
| `experiment_baseline.yaml` | experiment | The random-selection control on every PAM50 subtype-vs-rest task |
| `experiment_breastmnist.yaml` | experiment | The same design on BreastMNIST ResNet-50 embeddings (task 4), k scaled to 2,048 features |
| `experiment_scanb.yaml` | experiment | The full SCAN-B design: four selectors, three train fractions |
| `aircraft70_gpu.yaml` | experiment | FGVC-Aircraft, Boeing 737 vs A320 family: every selector on the GPU, k = 10 to 1000 and all features, 16 splits at 70% training |

## Dataset

**A dataset module**, such as
`featsel_datasets.scanb_gse96058.featsel_view`. It defines:

| Name | Meaning |
|------|---------|
| `NAME` | Name shown in the printed plan |
| `TARGET` | Default target |
| `load_xy(target=...)` | Returns `X, y`, with the module's choices of samples and genes |

The module must be importable, so run from the repo root.

**A CSV dataset config**, read by `DataLoader`. Keys it reads:

| Key | Meaning |
|-----|---------|
| `name` | Name shown in reports |
| `paths.features`, `paths.metadata` | CSV files, relative to this config |
| `sample_id_column` | Metadata column that identifies a sample |
| `target_column` | Default target |
| `transpose_features` | `true` if `features.csv` has samples as columns |
| `separator` | CSV separator, default `,` |

Other keys, such as `task`, `target_classes` and `alternative_targets`,
document the dataset for readers. The code does not read them. The runner
supports classification only.

## Experiment config

| Key | Default | Meaning |
|-----|---------|---------|
| `dataset` | required | A dataset module name, or a CSV dataset config (`.yaml`) relative to this file |
| `output` | required | Output CSV, relative to this file |
| `target` | dataset default | Target column to predict |
| `dataset_params` | `{}` | Further arguments to a dataset module's `load_xy`, such as `{drop_copies: true}` for BreastMNIST |
| `selectors` | required | List of `{name, k, params, label}`. `name` is a selector class name, case-insensitive, such as `ANOVAFSelector` |
| `models` | required | List of `{name, params, label, grid}`. `name` is one of `logistic_regression`, `linear_svm`, `random_forest`, `knn`, `lda` (and `xgboost` if installed). `grid`, such as `{C: [0.01, 0.1, 1, 10, 100]}`, tunes those parameters inside each split (see below) |
| `tuning` | `{n_splits: 3, train_size: 0.5, scoring: roc_auc}` | Inner splits for models with a `grid`. `scoring` is a scikit-learn scorer name |
| `seed` | `42` | Global seed; each split derives its own |
| `n_splits` | `100` | Stratified Monte Carlo splits |
| `selection_repeats` | `10` | Fits of each selection, each by a fresh selector. `selection_time_s` is their median; `selection_peak_mb` is the first fit's. `selection_stable` says whether every fit chose the same features. A selector that searches a hyperparameter by cross-validation, such as Lasso with `Cs` or RFECV, searches once first, timed as `search_time_s`; the repeats then fit with what it chose |
| `k_validation` | none | `{cv: 5}` adds `validation_roc_auc` to each row: the mean AUC of the row's k and model on `cv` folds of the training set, so k can be chosen without the test set (`featsel.chosen_k`). `validation_time_s` is the search's time per selection. Only selectors with more than one k; an entry opts out with `k_validation: false` |
| `train_sizes` | `[0.5]` | Training fractions |
| `preprocess` | `["standard"]` | `none`, `standard` or `quantile_normal`, fit on the training part |
| `metrics` | all but `roc_auc` | `accuracy`, `macro_f1`, `balanced_accuracy`, `mcc`, `g_mean` from the predicted labels; `roc_auc` from the model's continuous score (`decision_function`, else `predict_proba`) |
| `n_jobs` | `1` | Worker processes; splits run in parallel. `-1` uses every core |
| `max_worker_memory_gb` | `auto` | Cap on each worker's data memory: what it writes, not the shared split it maps. A cell past it is skipped and listed at the end, instead of the kernel killing a process; `--resume` fills it in later. `auto` shares the RAM free at the start among the workers, after room for the prepared splits. `null` sets no cap |

Every task is binary (`docs/SCOPE.md` 3a). A two-class target runs as one task,
`binary`. A target with more classes runs as one task per class, that class
vs the rest, named `ovr_<class>`.

Each value in a selector's `k` list is passed as its `n_features`. Four
selectors have no `n_features` and choose their own count: `FDRSelector`,
`FWESelector`, `PercentileSelector` and `RFECVSelector`. Leave out `k` for
them. A `k` fails at the first split. A `k` is a feature count for one
dataset, so it lives in the experiment config, not in the dataset module.

**Chains.** A selector entry may give `steps` instead of `name` and
`params`. Each step selects from the features the step before it kept, as
in a scikit-learn `Pipeline`. `k` goes to the last step; earlier steps set
their own count in `params`. The selection time covers every step. The
label defaults to the steps' class names joined by `+`.

```yaml
selectors:
  - label: MW1000+RFE
    k: [10, 50, 250]
    steps:
      - name: MannWhitneySelector
        params: {n_features: 1000}
      - name: RFESelector
        params: {C: 0.01}
```

`selected_indices` are always indices of the original features, and
`selector_params` holds the list of steps.

A model with a `grid` is tuned inside each split
(`final_report/03_how_to_measure.md`, 3.8). Inner stratified Monte Carlo splits of the
training half score every value in the grid. The model is refit on the whole
training half with the best values and scored once on the test half. The CSV
column `tuned_params` holds the chosen values, `{}` for an untuned model.
`fit_time_s` then includes the tuning.

## Running

```bash
python -m featsel.run --config configs/experiment_classifiers.yaml --n-jobs -1
```

`--out`, `--n-splits` and `--n-jobs` override the config. `--resume` skips
rows already in the output CSV. For a long unattended run, give the same
arguments to `scripts/run_until_done.py`: it restarts the runner with
`--resume` after a crash and keeps the machine awake
(`final_report/05_the_software.md`, section 5.5).

## Creating a new dataset

As a module (preferred when the data is downloaded or needs choices):

1. Create `featsel_datasets/<name>/` with the code that fetches the data and
   a `featsel_view.py` defining `NAME`, `TARGET` and `load_xy`.
2. Copy an experiment config, set its `dataset` to
   `featsel_datasets.<name>.featsel_view`, and set its `output`.

As CSV files:

1. Put `features.csv` and `metadata.csv` in `featsel_datasets/<name>/`.
2. Copy `template.yaml` to `<name>.yaml` and fill in the keys above.
3. Copy an experiment config, point its `dataset` at `<name>.yaml`, and set
   its `output`.
