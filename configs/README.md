# Configuration Files

An **experiment config** describes one comparison and names its dataset. The
dataset is either a Python module in `featsel_datasets/` (SCAN-B) or a
**dataset config** describing CSV files. Every path in a config is relative
to the file itself.

## Files

| File | Kind | Description |
|------|------|-------------|
| `template.yaml` | dataset | Documented template for a new CSV dataset config |
| `experiment_classifiers.yaml` | experiment | Random vs ANOVA F across five classifiers, 12 splits |
| `smoke_scanb.yaml` | experiment | Smoke test: every selector and classifier once on one SCAN-B split, about 3 minutes on 6 workers. A check, not a result |
| `experiment_baseline.yaml` | experiment | The random-selection control on every PAM50 subtype-vs-rest task |
| `experiment_breastmnist.yaml` | experiment | The same design on BreastMNIST ResNet-50 embeddings (task 4), k scaled to 2,048 features |
| `experiment_scanb.yaml` | experiment | The full SCAN-B design: four selectors, three train fractions |

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
| `selectors` | required | List of `{name, k, params, label}`. `name` is a selector class name, case-insensitive, such as `ANOVAFSelector` |
| `models` | required | List of `{name, params, label}`. `name` is one of `logistic_regression`, `linear_svm`, `random_forest`, `knn`, `lda` (and `xgboost` if installed) |
| `seed` | `42` | Global seed; each split derives its own |
| `n_splits` | `100` | Stratified Monte Carlo splits |
| `train_sizes` | `[0.5]` | Training fractions |
| `preprocess` | `["standard"]` | `none`, `standard` or `quantile_normal`, fit on the training part |
| `metrics` | all | `accuracy`, `macro_f1`, `balanced_accuracy`, `mcc`, `g_mean` |
| `n_jobs` | `1` | Worker processes; splits run in parallel. `-1` uses every core |
| `max_worker_memory_gb` | none | Cap on each worker's reserved memory. A cell past it is skipped and listed at the end, instead of the kernel killing a process; `--resume` fills it in later. An idle worker already reserves about 0.7 GB |

Every task is binary (`SCOPE.md` 3a). A two-class target runs as one task,
`binary`. A target with more classes runs as one task per class, that class
vs the rest, named `ovr_<class>`.

Each value in a selector's `k` list is passed as its `n_features`. Four
selectors have no `n_features` and choose their own count: `FDRSelector`,
`FWESelector`, `PercentileSelector` and `RFECVSelector`. Leave out `k` for
them. A `k` fails at the first split. A `k` is a feature count for one
dataset, so it lives in the experiment config, not in the dataset module.

## Running

```bash
python -m featsel.run --config configs/experiment_classifiers.yaml --n-jobs -1
```

`--out`, `--n-splits` and `--n-jobs` override the config. `--resume` skips
rows already in the output CSV.

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
