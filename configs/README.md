# Configuration Files

Two kinds of YAML file. A **dataset config** says where a dataset's files are
and how to read them. An **experiment config** names a dataset config and
describes one comparison. Every path in either kind is relative to the file
itself.

## Files

| File | Kind | Description |
|------|------|-------------|
| `template.yaml` | dataset | Documented template for a new CSV dataset config |
| `scanb_gse96058.yaml` | dataset | SCAN-B breast cancer from GSE96058: 2,976 patients (HiSeq 2000, one sample each), downloaded on first use |
| `experiment_classifiers.yaml` | experiment | Random vs ANOVA F across five classifiers, 12 splits |
| `experiment_baseline.yaml` | experiment | The random-selection control across all task framings |
| `experiment_scanb.yaml` | experiment | The full SCAN-B design: four selectors, three train fractions |

## Dataset config

A dataset config either names a loader function or describes CSV files.

**Loader.** The function returns `X, y`. The runner calls it with the
target and with `params`. Its module must be importable, so run from the
repo root.

| Key | Meaning |
|-----|---------|
| `name` | Name shown in reports |
| `loader` | Function as `module:function`, e.g. `featsel_datasets.scanb_gse96058.featsel_view:load_xy` |
| `target_column` | Default target |
| `params` | Keyword arguments passed to the function |

**CSV files.** Read by `DataLoader`. Keys it reads:

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
| `dataset` | required | Dataset config, relative to this file |
| `output` | required | Output CSV, relative to this file |
| `target` | dataset default | Target column to predict |
| `selectors` | required | List of `{name, k, params, label}`. `name` is a selector class name, case-insensitive, such as `ANOVAFSelector` |
| `models` | required | List of `{name, params, label}`. `name` is one of `logistic_regression`, `linear_svm`, `random_forest`, `knn`, `lda` (and `xgboost` if installed) |
| `seed` | `42` | Global seed; each split derives its own |
| `n_splits` | `100` | Stratified Monte Carlo splits |
| `train_sizes` | `[0.5]` | Training fractions |
| `preprocess` | `["standard"]` | `none`, `standard` or `quantile_normal`, fit on the training part |
| `task_framings` | `["multiclass"]` | `multiclass`, `one_vs_rest`, `one_vs_one` |
| `metrics` | all | `accuracy`, `macro_f1`, `balanced_accuracy`, `mcc`, `g_mean` |
| `n_jobs` | `1` | Worker processes; splits run in parallel. `-1` uses every core |

Each value in a selector's `k` list is passed as its `n_features`. Four
selectors have no `n_features` and choose their own count: `FDRSelector`,
`FWESelector`, `PercentileSelector` and `RFECVSelector`. Leave out `k` for
them. A `k` fails at the first split.

## Running

```bash
python -m featsel.run --config configs/experiment_classifiers.yaml --n-jobs -1
```

`--out`, `--n-splits` and `--n-jobs` override the config. `--resume` skips
rows already in the output CSV.

## Creating a new dataset

1. Put `features.csv` and `metadata.csv` in `featsel_datasets/<name>/`.
2. Copy `template.yaml` to `<name>.yaml` and fill in the keys above.
3. Copy an experiment config, point its `dataset` at `<name>.yaml`, and set
   its `output`.
