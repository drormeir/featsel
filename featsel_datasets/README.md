# Datasets

Each dataset should have its own subfolder with a corresponding config file in `configs/`.

## Expected Structure

```
featsel_datasets/
├── <dataset_name>/
│   ├── features.csv    # Feature matrix (genes/features x samples or samples x features)
│   └── metadata.csv    # Sample metadata with target labels
```

## Current Datasets

### SCAN-B Breast Cancer, GSE96058 (`scanb_gse96058/`)

The published source of both course versions below: 3,273 patients, 3,409
samples, 30,865 genes, from NCBI GEO accession GSE96058 (Brueffer et al.,
2018). `gse96058.py` downloads and caches the files on first use.
`featsel_view.py` returns `X, y` with this project's choices.
`demos/tutorial.ipynb` shows the content. Findings and decisions are in
`docs/report/03_data.md`.

### SCAN-B Breast Cancer, course version (`scanb_small/`)

Gene expression data from the Sweden Cancerome Analysis Network - Breast.

| File | Description |
|------|-------------|
| `features.csv` | Gene expression matrix (genes x samples) |
| `metadata.csv` | Sample metadata with PAM50 subtypes, ER status, survival data |

**Config**: `configs/scanb_small.yaml`

Shape: 9,264 genes x 3,069 samples. This is the pre-filtered version handed
out with the course and the one all reported results use. The gene filter is
not recorded, and it removed 11 of the 50 PAM50 genes. The 3,069 samples come
from 2,976 patients: 93 patients have two samples each.

### SCAN-B Breast Cancer, full gene set (`scanb_full/`)

Same samples, 30,865 genes x 3,069 samples (1.6 GB). Present as symlinks to a
copy outside the repo. Used only for scaling and timing experiments, not for
the reported method comparison. Its samples are the HiSeq 2000 samples of
`scanb_gse96058/`; values match on the 7 genes checked.

**Config**: `configs/scanb_full.yaml`

## Adding a New Dataset

1. Create a subfolder: `featsel_datasets/<your_dataset>/`
2. Add `features.csv` and `metadata.csv` following the expected format
3. Create a dataset config: `configs/<your_dataset>.yaml` (see `configs/README.md`)
4. Create an experiment config that names it, then run:
   `python -m featsel.run --config configs/<your_experiment>.yaml`
