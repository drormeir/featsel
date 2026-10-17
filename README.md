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

- Applies multiple feature selection methods: filter (variance, ANOVA F with a
  fixed count, a percentile, or a false discovery or family-wise error rate,
  mutual information, correlation), embedded (Lasso, random forest
  importance) and wrapper (recursive feature elimination, with or without
  cross-validation, and sequential selection), plus a random-selection
  control. Higher Criticism is planned.
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
├── featsel_datasets/   # One folder per dataset: loader code in git, data files not
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

Run a comparison from an experiment config. It names a dataset, the
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

To use your own dataset, add a folder in `featsel_datasets/` with either a
`featsel_view.py` module or CSV files plus a dataset config (see
`configs/template.yaml`). Then write an experiment config that names it (see
`configs/experiment_classifiers.yaml`). Paths in a config are relative to the
config file.

## Datasets

The pipeline is dataset-agnostic. Each dataset needs a folder in
`featsel_datasets/`. See `featsel_datasets/README.md`.

### SCAN-B Breast Cancer (GSE96058)

- Downloaded from NCBI GEO on first use (592 MB) and cached locally
- 2,976 patients (HiSeq 2000, one sample each), 28,552 genes for PAM50
- PAM50 molecular subtype labels (Basal, LumA, LumB, Her2, Normal)
- Clinical and pathology fields: ER, PgR, HER2, Ki67, grade, survival

Experiment configs name it as `dataset: "featsel_datasets.scanb_gse96058.featsel_view"`. Notes on the labels and on retired
course copies are in `docs/report/03_data.md`.

### BreastMNIST, ResNet-50 embeddings (second dataset)

- 780 breast ultrasound images (MedMNIST v2, 224 x 224), malignant vs benign or normal
- Each image is 2,048 ResNet-50 embedding dimensions: more features than images
- Images downloaded from Zenodo on first use (31 MB); embeddings computed once
  with PyTorch (`pip install -e ".[embeddings]"`) and cached

Config: `configs/experiment_breastmnist.yaml`. Notebooks in
`featsel_datasets/medmnist_breastmnist/demos/`.

## Status

This project is an M.Sc. final project at Reichman University, supervised by Dr. Ben Galili.
