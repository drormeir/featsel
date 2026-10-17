# Datasets

Each dataset has its own folder here. Git tracks the code and docs; it
ignores the data files (`*.csv`, `*.gz`).

## Current datasets

### SCAN-B Breast Cancer, GSE96058 (`scanb_gse96058/`)

3,273 patients, 3,409 samples, 30,865 genes, from NCBI GEO accession
GSE96058 (Brueffer et al., 2018).

- `gse96058.py` downloads and caches the published files on first use, and
  exposes them unchanged.
- `featsel_view.py` defines `NAME`, `TARGET` (`pam50 subtype`) and
  `load_xy`, which returns `X, y` with this project's choices: HiSeq 2000
  samples only, one sample per patient.
- `demos/tutorial.ipynb` shows the content.

Experiment configs name it as
`dataset: "featsel_datasets.scanb_gse96058.featsel_view"`. Findings and
decisions are in `docs/report/03_data.md`.

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

- **A module**, like `scanb_gse96058/`: a folder with code that downloads and
  caches the data, and a `featsel_view.py` defining `NAME`, `TARGET` and
  `load_xy(target=...)`. No dataset YAML.
- **CSV files**: `features.csv` and `metadata.csv` in the folder, read by
  `DataLoader`. Copy `configs/template.yaml` for the dataset config.

See `configs/README.md` for both forms. Then create an experiment config
whose `dataset` names the module or the dataset config, and run:
`python -m featsel.run --config configs/<your_experiment>.yaml`
