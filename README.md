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

- Applies multiple feature selection methods (filter, wrapper, and embedded approaches)
- Compares their effectiveness on classification tasks
- Scales efficiently through parallelization
- Produces interpretable results for domain experts

The primary use case is predicting breast cancer molecular subtypes from gene expression data, but the pipeline generalizes to other high-dimensional classification problems.

## Project Structure

```
├── configs/            # Dataset configuration files (YAML)
├── docs/               # Report chapters (Markdown)
├── notebooks/          # Jupyter notebooks for analysis
├── featsel/            # Main Python package
├── datasets/           # Input data (one subfolder per dataset)
├── figures/            # Generated plots
└── references/         # Project proposal and papers
```

## Installation

```bash
# Clone the repository
git clone https://github.com/drormeir/featsel.git
cd featsel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Future PyPI installation (not yet available):
```bash
pip install featsel
```

## Usage

Run the pipeline with a configuration file:

```bash
python -m featsel.run --config configs/scanb_small.yaml
```

To use your own dataset, create a config file (see `configs/scanb_small.yaml` as a template) and a data folder in `datasets/`.

## Datasets

The pipeline is dataset-agnostic. Each dataset needs:
- A subfolder in `datasets/` with `features.csv` and `metadata.csv`
- A YAML config file in `configs/`

### SCAN-B Breast Cancer (included config)

- Gene expression measurements (thousands of features)
- PAM50 molecular subtype labels (Basal, LumA, LumB, HER2, Normal)
- Clinical metadata (ER status, survival data)

**Note**: Data files are not included due to size. Download from [TBD] and place in `datasets/scanb_small/`.

## Status

This project is part of an M.Sc. thesis at Reichman University, supervised by Dr. Ben Galili.
