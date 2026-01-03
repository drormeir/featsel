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

### SCAN-B Breast Cancer (`scanb/`)

Gene expression data from the Sweden Cancerome Analysis Network - Breast.

| File | Description |
|------|-------------|
| `features.csv` | Gene expression matrix (genes x samples) |
| `metadata.csv` | Sample metadata with PAM50 subtypes, ER status, survival data |

**Config**: `configs/scanb.yaml`

**Download**: [TBD - link to be added]

## Adding a New Dataset

1. Create a subfolder: `datasets/<your_dataset>/`
2. Add `features.csv` and `metadata.csv` following the expected format
3. Create a config file: `configs/<your_dataset>.yaml`
4. Run the pipeline with: `python -m src.run --config configs/<your_dataset>.yaml`
