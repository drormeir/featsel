# Datasets

Each dataset has its own folder here and a dataset config in `configs/`.
Git tracks the code and docs; it ignores the data files (`*.csv`, `*.gz`).

## Current datasets

### SCAN-B Breast Cancer, GSE96058 (`scanb_gse96058/`)

3,273 patients, 3,409 samples, 30,865 genes, from NCBI GEO accession
GSE96058 (Brueffer et al., 2018).

- `gse96058.py` downloads and caches the published files on first use, and
  exposes them unchanged.
- `featsel_view.py` returns `X, y` with this project's choices: HiSeq 2000
  samples only, one sample per patient.
- `demos/tutorial.ipynb` shows the content.

**Config**: `configs/scanb_gse96058.yaml`. Findings and decisions are in
`docs/report/03_data.md`.

## Retired datasets

`scanb_small` and `scanb_full`, the course's copies of SCAN-B, were retired
on 2026-10-17. Both hold the 3,069 HiSeq 2000 samples of GSE96058, which come
from 2,976 patients: 93 patients have two samples each. `scanb_small` keeps
9,264 genes chosen by an unrecorded rule, and lacks 11 of the 50 PAM50 genes.
Results computed before 2026-10-17 used `scanb_small`.

The data and their configs are archived, outside git, in
`references/local/scanb_course/`.

## Adding a new dataset

Either:

- **A loader**, like `scanb_gse96058/`: a folder with code that downloads and
  caches the data, and a function that returns `X, y`. Its dataset config
  names the function under `loader`.
- **CSV files**: `features.csv` and `metadata.csv` in the folder, read by
  `DataLoader`. Copy `configs/template.yaml` for the dataset config.

See `configs/README.md` for both config forms. Then create an experiment
config that names the dataset config, and run:
`python -m featsel.run --config configs/<your_experiment>.yaml`
