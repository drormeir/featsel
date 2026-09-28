# Configuration Files

Two kinds of YAML file. A **dataset config** says where a dataset's files are
and how to read them. An **experiment config** names a dataset config and
describes one comparison. Every path in either kind is relative to the file
itself.

## Files

| File | Kind | Description |
|------|------|-------------|
| `template.yaml` | dataset | Documented template for a new dataset config |
| `scanb_small.yaml` | dataset | SCAN-B breast cancer, course version: 3,069 patients, 9,259 genes after cleaning |
| `scanb_full.yaml` | dataset | SCAN-B breast cancer, full gene set, for timing runs only |
| `experiment_classifiers.yaml` | experiment | Random vs ANOVA F across five classifiers, 12 splits |
| `experiment_baseline.yaml` | experiment | The random-selection control across all task framings |
| `experiment_scanb.yaml` | experiment | The full SCAN-B design: four selectors, three train fractions |

## Dataset config

Keys the loader reads:

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

## Running

```bash
python -m featsel.run --config configs/experiment_classifiers.yaml --n-jobs -1
```

`--out`, `--n-splits` and `--n-jobs` override the config. `--resume` skips
rows already in the output CSV.

## Creating a new dataset

1. Put `features.csv` and `metadata.csv` in `datasets/<name>/`.
2. Copy `template.yaml` to `<name>.yaml` and fill in the keys above.
3. Copy an experiment config, point its `dataset` at `<name>.yaml`, and set
   its `output`.
