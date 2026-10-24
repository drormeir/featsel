# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `SCOPE.md` first.** It defines what is allowed to be built and how to work with Dror. It overrides anything here that looks like an invitation to expand the project (PyPI publishing, future phases, PyTorch integration).

## Project Overview

`featsel` is a feature selection pipeline for high-dimensional data, focused on genomics and bioinformatics. The primary use case is predicting breast cancer molecular subtypes from gene expression data (thousands of features, relatively few samples). The project is an M.Sc. final project at Reichman University.

## Development Commands

### Environment Setup
```bash
# Create and activate the virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with dev tools (same as -e ".[dev]")
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# Run an experiment (from the repo root, so featsel_datasets imports)
python -m featsel.run --config configs/experiment_scanb.yaml

# Load SCAN-B; the first call downloads GSE96058 (592 MB) into the cache
python -c "from featsel_datasets.scanb_gse96058.featsel_view import load_xy; X, y = load_xy(); print(X.shape)"

# The second dataset, BreastMNIST embeddings; the first call needs the
# 'embeddings' extra (pip install -e ".[embeddings]") to compute them
python -m featsel.run --config configs/experiment_breastmnist.yaml --n-jobs -1
```

### Feature Selection
```bash
# Use a selector class in Python scripts or notebooks
python
>>> from featsel.selectors import ANOVAFSelector
>>> from featsel_datasets.scanb_gse96058.featsel_view import load_xy
>>> X, y = load_xy()
>>> selector = ANOVAFSelector(n_features=100)
>>> selector.fit(X, y)
>>> X_selected = selector.transform(X)

# Run one selector's tests
python -m pytest featsel/selectors/anova_f -v

# Run all tests (tests/ and the selector tests under featsel/)
python -m pytest -v
```

### Package Building and Publishing
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build package
python -m build

# Verify build
twine check dist/*

# Upload to TestPyPI (for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

### Code Quality (Optional Dependencies)
```bash
# Format code
black featsel/

# Lint code
flake8 featsel/

# Type checking
mypy featsel/
```

## Architecture

### Data Loading System

A dataset is loaded one of two ways. Both give `X, y` to the runner.

**Dataset loaders: `featsel_datasets/<name>/`** (used for SCAN-B and BreastMNIST)

A folder per dataset with code that downloads the published files once and
caches them beside it. For SCAN-B (`featsel_datasets/scanb_gse96058/`):
- `gse96058.py`: class `GSE96058`, exposes every sample, gene and field of
  GEO accession GSE96058 unchanged.
- `featsel_view.py`: `NAME`, `TARGET` and `load_xy()`, this project's
  choices (HiSeq 2000 only, one sample per patient, chosen target, no
  single-value genes). `load_xy` returns a DataFrame and a Series. An
  experiment config names this module as its `dataset`.
- `demos/tutorial.ipynb`: what the dataset contains.

BreastMNIST (`featsel_datasets/medmnist_breastmnist/`) has the same shape:
`breastmnist.py` (class `BreastMNIST`: images, labels, and ResNet-50
embeddings computed once with PyTorch and cached) and `featsel_view.py`.

`featsel_datasets` is a regular package (with `__init__.py` files) but is not
installed: `pyproject.toml` excludes it from the `featsel` package. Run from
the repo root.
Design: `docs/architecture_options.md`, "Dataset loaders". Data findings:
`docs/report/03_data.md`.

**CSV datasets: `featsel/data_loader.py`**

The `DataLoader` class:
- Loads datasets from YAML configuration files in `configs/`
- Expects two CSV files per dataset: `features.csv` (high-dimensional feature matrix) and `metadata.csv` (sample labels)
- Automatically aligns samples between features and metadata by index
- Cleans data by removing: (1) rows/columns that are entirely NaN, (2) features with only one unique value
- Supports multiple target variables in the same dataset via `set_target()`
- Provides PyTorch-style indexing: `loader[idx]` returns `(X, y)` tuples
- Generates a detailed loading report tracking samples/features before and after cleaning

**Configuration System: `configs/`**

An experiment config's `dataset` is either a dataset module (e.g.
`featsel_datasets.scanb_gse96058.featsel_view`, no YAML) or a CSV dataset
config. A CSV dataset config specifies:
- Paths to `features.csv` and `metadata.csv` (relative to the config file)
- `sample_id_column`: Column name to use as sample identifier
- `target_column`: Default target variable for prediction
- `transpose_features`: Whether to transpose feature matrix (if samples are columns instead of rows)
- Task type: classification (binary/multiclass) or regression
- Optional alternative targets available in the metadata

`configs/template.yaml` documents the CSV form.

### Dataset Structure

**CSV format in `featsel_datasets/<dataset_name>/`:**
- `features.csv`: Feature matrix with samples as rows and features (genes) as columns, OR transposed (set `transpose_features: true` in config)
- `metadata.csv`: Sample metadata with target labels and additional clinical variables

**Current dataset: SCAN-B, GSE96058 (`featsel_datasets/scanb_gse96058/`)**
- 3,273 patients, 3,409 samples, 30,865 genes; `load_xy()` gives 2,976 patients
- Default target `pam50 subtype` (Basal, LumA, LumB, Her2, Normal)
- Every task is binary (`SCOPE.md` 3a): PAM50 becomes five tasks, each subtype vs the rest. No five-class results
- Other targets are GSE96058's own field names, e.g. `er status` (pathologist), `er prediction mgc` (computed from genes), `overall survival event`
- PAM50 and the `prediction` fields are computed from gene expression, not measured. See `docs/report/03_data.md`.

**Second dataset (task 4): BreastMNIST (`featsel_datasets/medmnist_breastmnist/`)**
- 780 breast ultrasound images (MedMNIST v2, 224 x 224), malignant 210 vs benign or normal 570
- Features: 2,048 ResNet-50 embedding dimensions per image (ImageNet weights, last layer removed)
- Target `label`; all 780 images are used, MedMNIST's own split is ignored
- Computing embeddings needs `torch`/`torchvision` (the `embeddings` extra); they are cached as `embeddings_resnet50.npz`

The course copies `scanb_small` and `scanb_full` are retired; they are
archived outside git in `references/local/scanb_course/`.

### Package Structure

```
featsel/
├── __init__.py              # Exports DataLoader, FeatureSelector, summarize, kuncheva_index
├── data_loader.py           # Core data loading and cleaning logic
├── run.py                   # Config-driven experiment runner: run(X, y, config), CLI
├── analysis.py              # summarize() and kuncheva_index() over the runner's CSV
├── metrics.py               # Predictive metrics and the CSV's key columns
├── strength.py              # loo_confusion(): nested leave-one-out score of chosen genes
├── selectors/               # Feature selection methods, one subpackage each:
│   │                        #   __init__.py (the class), tests/, demos/tutorial.ipynb
│   ├── __init__.py          # Imports every selector, which registers it
│   ├── base.py              # FeatureSelector: scikit-learn base, registration, create()
│   ├── anova_f/             # ANOVAFSelector
│   ├── correlation/         # CorrelationSelector
│   ├── fdr/                 # FDRSelector
│   ├── fwe/                 # FWESelector
│   ├── higher_criticism/    # HigherCriticismSelector; log_pvalue.py: F-test log p below 1e-308
│   ├── lasso/               # LassoSelector
│   ├── mutual_info/         # MutualInfoSelector
│   ├── percentile/          # PercentileSelector
│   ├── random_selection/    # RandomSelector (not `random`: shadows the stdlib)
│   ├── rfe/                 # RFESelector
│   ├── rfecv/               # RFECVSelector
│   ├── sequential/          # SequentialSelector
│   ├── tree_importance/     # TreeImportanceSelector
│   └── variance_threshold/  # VarianceThreshold
└── utils/
    ├── __init__.py
    └── validation.py        # Input validation helpers (currently unused)
```

Every selector subclasses `FeatureSelector`, which inherits scikit-learn's
`BaseEstimator` and `SelectorMixin`. A selector lists its parameters in
`__init__` (no `**kwargs`) and sets `support_` in `fit`. It therefore works
with `Pipeline`, `clone`, `cross_val_score` and `GridSearchCV`, and a
misspelled parameter raises. As in scikit-learn, `transform` returns a NumPy
array unless `set_output(transform='pandas')` is set.

Selectors register themselves by class name. In Python, import the class.
`FeatureSelector.create(name, **params)` is for names read as text, such as the
runner's YAML configs, which name selectors by class name (case-insensitive).
See `docs/architecture_options.md`, section "Selector design".

Built: VarianceThreshold, ANOVAFSelector, MutualInfoSelector,
CorrelationSelector, FDRSelector, FWESelector, PercentileSelector,
LassoSelector, TreeImportanceSelector, RFESelector, RFECVSelector,
SequentialSelector, RandomSelector, HigherCriticismSelector.

Every selector that wraps a scikit-learn method has two tests next to it:
it selects the same features as that method, and `fit` plus `get_support` is
at most 5% slower. The shared checks are in `tests/sklearn_reference.py`.

## Feature Selection Usage

### Basic sklearn Pipeline Integration

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from featsel.selectors import ANOVAFSelector
from featsel_datasets.scanb_gse96058.featsel_view import load_xy

X, y = load_xy()

pipe = Pipeline([
    ('select', ANOVAFSelector(n_features=100)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Selection is refit inside every fold
scores = cross_val_score(pipe, X, y, cv=5)

# Selected features and their scores
pipe.fit(X, y)
select = pipe.named_steps['select']
selected = select.get_feature_names_out()
scores_per_feature = select.feature_importances_
```

### Available Selectors

Each has a tutorial in `featsel/selectors/<name>/demos/tutorial.ipynb`.

```python
from featsel.selectors import (
    ANOVAFSelector, CorrelationSelector, FDRSelector, FWESelector,
    HigherCriticismSelector, LassoSelector, MutualInfoSelector, PercentileSelector, RandomSelector,
    RFECVSelector, RFESelector, SequentialSelector, TreeImportanceSelector,
    VarianceThreshold,
)

# Filters
VarianceThreshold(threshold=0.01)                        # drop near-constant features
ANOVAFSelector(n_features=100)                           # univariate F-test
FDRSelector(alpha=0.05)                                  # F-test, Benjamini-Hochberg; own count
FWESelector(alpha=0.05)                                  # F-test, Bonferroni; own count
PercentileSelector(percentile=1)                         # F-test, top 1%
HigherCriticismSelector(variant='2008')                  # F-test, HC threshold; own count; or '2004'
MutualInfoSelector(n_features=100, random_state=42)      # any dependence, slower
CorrelationSelector(n_features=100, target_threshold=0.1,
                    inter_feature_threshold=0.95)        # drops redundant pairs
# Embedded
LassoSelector(n_features=None, C=0.1, random_state=42)   # L1 model, picks its own count
TreeImportanceSelector(n_features=100, random_state=42)  # random forest importance
# Wrappers
RFESelector(n_features=100, C=0.01, random_state=42)     # SVM-RFE; small C when p >> n
RFECVSelector(cv=5, C=0.01, random_state=42)             # SVM-RFE, count chosen by inner CV
SequentialSelector(n_features=10, cv=5)                  # greedy forward; slow, prefilter first
# Control
RandomSelector(n_features=100, random_state=42)          # the control
```

Every selector takes `check_input=True`. `False` skips input validation in
`fit`; use it only for a clean NumPy array.

### High-Dimensional Use Case (few samples, many features)

For transfer learning with ResNet embeddings where embedding dimension >> sample count:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from featsel.selectors import ANOVAFSelector, MutualInfoSelector, VarianceThreshold

# Example: 500 ResNet features, 50 rare disease CT image samples

# Step 1: Quick preprocessing - remove zero-variance features
prefilter = VarianceThreshold(threshold=0.01)

# Step 2: Fast univariate screening to reduce dimensionality
quick_select = ANOVAFSelector(n_features=200)

# Step 3: Final selection
final_select = MutualInfoSelector(n_features=50, random_state=42)

# Create pipeline
pipe = Pipeline([
    ('prefilter', prefilter),
    ('quick_select', quick_select),
    ('final_select', final_select),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train on embeddings
pipe.fit(X_embeddings, y_disease_labels)
```

### Multiple Targets

```python
from featsel_datasets.scanb_gse96058.featsel_view import load_xy
from featsel_datasets.scanb_gse96058.gse96058 import GSE96058

data = GSE96058()                                   # load once, reuse
X, y_pam50 = load_xy(data, target='pam50 subtype')
X_er, y_er = load_xy(data, target='er status')      # pathologist ER; drops patients without it
```

A CSV dataset switches targets with `DataLoader.set_target()`.

## Important Notes

### Data Files
- Data files are not tracked in git: `.gitignore` ignores `*.csv`, `*.gz`, `*.npz`, `*.part` and `*.txt` anywhere, and a file that must be tracked is named as an exception (e.g. `!requirements.txt`); the loader code is tracked
- GSE96058's gene expression file is 592 MB; loading it takes about a minute
- BreastMNIST's images are 31 MB; its cached embeddings 4 MB
- When working with data loading, test on a subset first to avoid long load times

### Configuration Files
- YAML configs in `configs/` drive all pipeline behavior
- `template.yaml` documents the CSV dataset form; `configs/README.md` documents both forms
- Every path in a config is relative to that config file: CSV dataset configs use `../featsel_datasets/...`, experiment configs name their output `../results/...` and a CSV dataset config the same way. A dataset module is named by its import path, not a file path.

### Package Installation
- The package uses modern Python packaging with `pyproject.toml`
- Designed for PyPI distribution but currently in development (version 0.1.0)
- Support for Python 3.9-3.13
- Core dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, pyyaml, tqdm
- Optional dependencies: jupyter (dev), optuna (hyperparameter optimization), torch and torchvision (embeddings: only to compute the BreastMNIST embeddings)

### Index Alignment
- The DataLoader automatically handles mismatches between feature and metadata sample indices
- It only keeps samples present in BOTH files (inner join behavior)
- Reports track how many samples were dropped during alignment

