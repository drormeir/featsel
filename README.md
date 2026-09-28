# featsel

Feature selection pipeline for high-dimensional data with a focus on genomics and bioinformatics.

## The Problem

Modern datasets often contain thousands of features but relatively few samples. This is especially common in:

- **Genomics**: Gene expression profiles with 20,000+ genes per patient
- **Text analysis**: Document classification with large vocabularies
- **Sensor data**: IoT and industrial monitoring systems

Training models on such data leads to overfitting, long computation times, and poor interpretability. Feature selection addresses this by identifying the most predictive variables while discarding noise.

## What This Project Does

This project implements a feature selection pipeline that:

- Applies multiple feature selection methods: filter (variance, ANOVA F, mutual
  information, correlation) and embedded (Lasso, random forest importance),
  plus a random-selection control. Wrapper (RFE) and Higher Criticism are planned.
- Compares them under one protocol: stratified Monte Carlo splits, with every
  selector fit on the training part only
- Runs the splits in parallel
- Records accuracy metrics, selection stability, selection time and memory

Every selector is a scikit-learn estimator, so it also works on its own in a
scikit-learn `Pipeline`, `cross_val_score` or `GridSearchCV`.

The primary use case is predicting breast cancer molecular subtypes from gene expression data, but the pipeline generalizes to other high-dimensional classification problems.

## Project Structure

```
├── featsel/            # The Python package
│   └── selectors/      # One subpackage per selector, with its tests and a tutorial notebook
├── configs/            # Dataset and experiment configuration files (YAML)
├── datasets/           # Input data (one subfolder per dataset, not in git)
├── results/            # Experiment output CSVs (not in git)
├── figures/            # Generated plots
├── notebooks/          # Analysis and usage notebooks
├── scripts/            # One-off runs and examples
├── tests/              # Package-level tests
├── docs/               # Design notes; report chapters in docs/report/
└── references/         # Project proposal and papers
```

## Installation

```bash
# Clone the repository
git clone https://github.com/drormeir/featsel.git
cd featsel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Future PyPI installation (not yet available):
```bash
pip install featsel
```

## Usage

Run a comparison from an experiment config. It names a dataset config, the
selectors, the models and the splits, and writes one CSV row per execution:

```bash
python -m featsel.run --config configs/experiment_classifiers.yaml --n-jobs -1
python -m featsel.run --config configs/experiment_classifiers.yaml --resume   # continue a killed run
```

The same from Python, on data already in memory:

```python
from featsel.run import run

config = {
    'selectors': [{'name': 'RandomSelector', 'k': [10, 50]},
                  {'name': 'ANOVAFSelector', 'k': [10, 50]}],
    'models': [{'name': 'logistic_regression', 'params': {'max_iter': 1000}}],
    'n_splits': 20,
    'output': 'results/my_run.csv',
}
run(X, y, config)
```

A single selector, as a scikit-learn step:

```python
from featsel.selectors import ANOVAFSelector

X_selected = ANOVAFSelector(n_features=50).fit_transform(X, y)
```

See `notebooks/03_usage_demo.ipynb` for an end-to-end example.

To use your own dataset, add a data folder in `datasets/`, a dataset config
(see `configs/template.yaml`) and an experiment config that names it (see
`configs/experiment_classifiers.yaml`). Paths in a config are relative to the
config file.

## Datasets

The pipeline is dataset-agnostic. Each dataset needs:
- A subfolder in `datasets/` with `features.csv` and `metadata.csv`
- A dataset config file in `configs/`

### SCAN-B Breast Cancer (included config)

- 3,069 patients, 9,259 gene expression features after cleaning
- PAM50 molecular subtype labels (Basal, LumA, LumB, HER2, Normal)
- Clinical metadata (ER status, survival data)

**Note**: Data files are not included due to size. Download from [TBD] and place in `datasets/scanb_small/`.

## Status

This project is an M.Sc. final project at Reichman University, supervised by Dr. Ben Galili.
