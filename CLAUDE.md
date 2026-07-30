# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `SCOPE.md` first.** It defines what is allowed to be built and how to work with Dror. It overrides anything here that looks like an invitation to expand the project (PyPI publishing, future phases, PyTorch integration).

## Project Overview

`featsel` is a feature selection pipeline for high-dimensional data, focused on genomics and bioinformatics. The primary use case is predicting breast cancer molecular subtypes from gene expression data (thousands of features, relatively few samples). The project is part of an M.Sc. thesis at Reichman University.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install with development dependencies
pip install -e ".[dev]"
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
# Use FeatureSelector in Python scripts or notebooks
python
>>> from featsel import DataLoader, FeatureSelector
>>> loader = DataLoader('configs/scanb_small.yaml')
>>> selector = FeatureSelector(method='anova_f', n_features=100)
>>> selector.fit(loader.X, loader.y)
>>> X_selected = selector.transform(loader.X)

# Run tests for feature selection
python -m pytest tests/test_feature_selector.py -v

# Run all tests
python -m pytest tests/ -v
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
- Paths to `features.csv` and `metadata.csv` (relative to project root)
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
├── __init__.py              # Exports DataLoader, FeatureSelector
├── data_loader.py           # Core data loading and cleaning logic
├── feature_selector.py      # Main sklearn-compatible API
├── selectors/               # Feature selection methods
│   ├── __init__.py          # Exports all selectors
│   ├── base.py              # BaseSelector abstract class
│   ├── filter.py            # Filter methods (variance, ANOVA, mutual info, correlation)
│   ├── embedded.py          # Embedded methods (Lasso, trees) - Phase 2
│   └── wrapper.py           # Wrapper methods (RFE) - Phase 3
└── utils/                   # Utility functions
    ├── __init__.py
    ├── validation.py        # Input validation
    └── reporting.py         # Feature importance reports - Phase 2
```

**Current Status (Phase 1 - COMPLETE):**
- ✅ BaseSelector abstract class
- ✅ Filter methods: VarianceThreshold, ANOVAFSelector, MutualInfoSelector, CorrelationSelector
- ✅ FeatureSelector main API with sklearn TransformerMixin
- ✅ Full sklearn Pipeline integration
- ✅ Comprehensive test suite (26 tests, 100% passing)

**Future development:**
- Phase 2: Embedded methods (Lasso, ElasticNet, tree-based importance)
- Phase 3: Wrapper methods (RFE)
- Phase 4: PyTorch Dataset integration
- Model evaluation pipelines
- Visualization utilities

## Feature Selection Usage

### Basic sklearn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from featsel import DataLoader, FeatureSelector

# Load data
loader = DataLoader('configs/scanb_small.yaml')

# Create pipeline with feature selection
pipe = Pipeline([
    ('select', FeatureSelector(method='anova_f', n_features=100)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train and evaluate
pipe.fit(loader.X, loader.y)
score = pipe.score(X_test, y_test)

# Access selected features
selected_features = pipe.named_steps['select'].selected_features_
print(f"Selected {len(selected_features)} features: {selected_features[:10]}")

# Get feature importance report
report = pipe.named_steps['select'].get_report()
report.to_csv('feature_selection_report.csv')
```

### Available Filter Methods

1. **VarianceThreshold** - Remove low/zero variance features (fastest)
```python
selector = FeatureSelector(method='variance_threshold', threshold=0.01)
```

2. **ANOVAFSelector** - ANOVA F-test for univariate feature selection
```python
selector = FeatureSelector(method='anova_f', n_features=100, task='classification')
```

3. **MutualInfoSelector** - Mutual information (captures non-linear relationships)
```python
selector = FeatureSelector(method='mutual_info', n_features=100, n_neighbors=3, random_state=42)
```

4. **CorrelationSelector** - Correlation-based selection with redundancy removal
```python
selector = FeatureSelector(
    method='correlation',
    n_features=100,
    target_threshold=0.1,
    inter_feature_threshold=0.95
)
```

### High-Dimensional Use Case (few samples, many features)

For transfer learning with ResNet embeddings where embedding dimension >> sample count:

```python
from featsel import FeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Example: 500 ResNet features, 50 rare disease CT image samples

# Step 1: Quick preprocessing - remove zero-variance features
prefilter = FeatureSelector(method='variance_threshold', threshold=0.01)

# Step 2: Fast univariate screening to reduce dimensionality
quick_select = FeatureSelector(method='anova_f', n_features=200)

# Step 3: Final selection (can use Lasso in Phase 2 for better results)
final_select = FeatureSelector(method='mutual_info', n_features=50, random_state=42)

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
from featsel import DataLoader, FeatureSelector

loader = DataLoader('configs/scanb_small.yaml')

# Fit feature selection on PAM50 target
loader.set_target('PAM50')
selector = FeatureSelector(method='anova_f', n_features=100)
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
- Always validate paths are relative to the project root, not the config directory

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

