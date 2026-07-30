# Datasets

Each dataset should have its own subfolder with a corresponding config file in `configs/`.

## Expected Structure

```
datasets/
├── <dataset_name>/
│   ├── features.csv    # Feature matrix (genes/features x samples or samples x features)
│   └── metadata.csv    # Sample metadata with target labels
```

## Current Datasets

### SCAN-B Breast Cancer, course version (`scanb_small/`)

Gene expression data from the Sweden Cancerome Analysis Network - Breast.

| File | Description |
|------|-------------|
| `features.csv` | Gene expression matrix (genes x samples) |
| `metadata.csv` | Sample metadata with PAM50 subtypes, ER status, survival data |

**Config**: `configs/scanb_small.yaml`

Shape: 9,265 genes x 3,069 samples. This is the pre-filtered version handed out
with the course and the one all reported results use.

### SCAN-B Breast Cancer, full gene set (`scanb_full/`)

Same cohort, 30,866 genes x 3,069 samples (1.6 GB). Present as symlinks to a copy
outside the repo. Used only for scaling and timing experiments, not for the
reported method comparison.

**Config**: `configs/scanb_full.yaml`

**Download**: [TBD - link to be added]

## Adding a New Dataset

1. Create a subfolder: `datasets/<your_dataset>/`
2. Add `features.csv` and `metadata.csv` following the expected format
3. Create a config file: `configs/<your_dataset>.yaml`
4. Run the pipeline with: `python -m src.run --config configs/<your_dataset>.yaml`
