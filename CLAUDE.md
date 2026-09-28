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
# Run with a configuration file
python -m featsel.data_loader configs/scanb_small.yaml

# Test the DataLoader directly
python featsel/data_loader.py configs/scanb_small.yaml
```

### Feature Selection
```bash
# Use a selector class in Python scripts or notebooks
python
>>> from featsel import DataLoader
>>> from featsel.selectors import ANOVAFSelector
>>> loader = DataLoader('configs/scanb_small.yaml')
>>> selector = ANOVAFSelector(n_features=100)
>>> selector.fit(loader.X, loader.y)
>>> X_selected = selector.transform(loader.X)

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

The core architecture revolves around a PyTorch-style DataLoader that provides a consistent interface for loading high-dimensional datasets.

**Key Component: `featsel/data_loader.py`**

The `DataLoader` class:
- Loads datasets from YAML configuration files in `configs/`
- Expects two CSV files per dataset: `features.csv` (high-dimensional feature matrix) and `metadata.csv` (sample labels)
- Automatically aligns samples between features and metadata by index
- Cleans data by removing: (1) rows/columns that are entirely NaN, (2) features with only one unique value
- Supports multiple target variables in the same dataset via `set_target()`
- Provides PyTorch-style indexing: `loader[idx]` returns `(X, y)` tuples
- Generates a detailed loading report tracking samples/features before and after cleaning

**Configuration System: `configs/`**

Each dataset requires a YAML config file specifying:
- Paths to `features.csv` and `metadata.csv` (relative to the config file)
- `sample_id_column`: Column name to use as sample identifier
- `target_column`: Default target variable for prediction
- `transpose_features`: Whether to transpose feature matrix (if samples are columns instead of rows)
- Task type: classification (binary/multiclass) or regression
- Optional alternative targets available in the metadata

Example: `configs/scanb_small.yaml` configures the SCAN-B breast cancer dataset with PAM50 subtypes as the primary target, and ER status and survival data as alternative targets.

### Dataset Structure

**Expected format in `datasets/<dataset_name>/`:**
- `features.csv`: Feature matrix with samples as rows and features (genes) as columns, OR transposed (set `transpose_features: true` in config)
- `metadata.csv`: Sample metadata with target labels and additional clinical variables

**Current dataset: `datasets/scanb_small/`**
- 518MB gene expression matrix with thousands of genes per sample
- PAM50 molecular subtypes (Basal, LumA, LumB, Her2, Normal) as primary classification target
- Alternative targets: ER status (binary), survival event (binary), survival time (regression)

### Package Structure

```
featsel/
├── __init__.py              # Exports DataLoader, FeatureSelector, summarize, kuncheva_index
├── data_loader.py           # Core data loading and cleaning logic
├── run.py                   # Config-driven experiment runner: run(X, y, config), CLI
├── analysis.py              # summarize() and kuncheva_index() over the runner's CSV
├── metrics.py               # Predictive metrics and the CSV's key columns
├── selectors/               # Feature selection methods, one subpackage each:
│   │                        #   __init__.py (the class), tests/, demos/tutorial.ipynb
│   ├── __init__.py          # Imports every selector, which registers it
│   ├── base.py              # FeatureSelector: scikit-learn base, registration, create()
│   ├── anova_f/             # ANOVAFSelector
│   ├── correlation/         # CorrelationSelector
│   ├── lasso/               # LassoSelector
│   ├── mutual_info/         # MutualInfoSelector
│   ├── random_selection/    # RandomSelector (not `random`: shadows the stdlib)
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
CorrelationSelector, LassoSelector, TreeImportanceSelector, RandomSelector.
Planned (see `NEXT.md`): RFESelector, HigherCriticismSelector.

## Feature Selection Usage

### Basic sklearn Pipeline Integration

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from featsel import DataLoader
from featsel.selectors import ANOVAFSelector

loader = DataLoader('configs/scanb_small.yaml')

pipe = Pipeline([
    ('select', ANOVAFSelector(n_features=100)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Selection is refit inside every fold
scores = cross_val_score(pipe, loader.X, loader.y, cv=5)

# Selected features and their scores
pipe.fit(loader.X, loader.y)
select = pipe.named_steps['select']
selected = select.get_feature_names_out()
scores_per_feature = select.feature_importances_
```

### Available Selectors

Each has a tutorial in `featsel/selectors/<name>/demos/tutorial.ipynb`.

```python
from featsel.selectors import (
    ANOVAFSelector, CorrelationSelector, LassoSelector, MutualInfoSelector,
    RandomSelector, TreeImportanceSelector, VarianceThreshold,
)

VarianceThreshold(threshold=0.01)                        # drop near-constant features
ANOVAFSelector(n_features=100)                           # univariate F-test
MutualInfoSelector(n_features=100, random_state=42)      # any dependence, slower
CorrelationSelector(n_features=100, target_threshold=0.1,
                    inter_feature_threshold=0.95)        # drops redundant pairs
LassoSelector(n_features=None, C=0.1, random_state=42)   # L1 model, picks its own count
TreeImportanceSelector(n_features=100, random_state=42)  # random forest importance
RandomSelector(n_features=100, random_state=42)          # the control
```

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

### Multi-Target Support with DataLoader

```python
from featsel import DataLoader
from featsel.selectors import ANOVAFSelector

loader = DataLoader('configs/scanb_small.yaml')

# Fit feature selection on PAM50 target
loader.set_target('PAM50')
selector = ANOVAFSelector(n_features=100)
selector.fit(loader.X, loader.y)

# Transform works with any target
X_selected = selector.transform(loader.X)

# Switch to different target (ER status)
loader.set_target('ER')
# Same selected features, different target for modeling
```

## Important Notes

### Data Files
- Data files in `datasets/scanb_small/` are not tracked in git (large files, potentially sensitive)
- The `features.csv` file is ~519MB, containing full gene expression matrix
- When working with data loading, test on a subset first to avoid long load times

### Configuration Files
- YAML configs in `configs/` drive all pipeline behavior
- `template.yaml` contains full documentation of all available options
- Every path in a config is relative to that config file: dataset configs use `../datasets/...`, experiment configs name their dataset config and `../results/...` output the same way

### Package Installation
- The package uses modern Python packaging with `pyproject.toml`
- Designed for PyPI distribution but currently in development (version 0.1.0)
- Support for Python 3.9-3.13
- Core dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, pyyaml, tqdm
- Optional dependencies: jupyter (dev), optuna (hyperparameter optimization)

### Index Alignment
- The DataLoader automatically handles mismatches between feature and metadata sample indices
- It only keeps samples present in BOTH files (inner join behavior)
- Reports track how many samples were dropped during alignment

