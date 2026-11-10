# Datasets

Each dataset has its own folder here. Git tracks the code and docs; it
ignores data files (`*.csv`, `*.gz`, `*.npz`, `*.txt`) anywhere in the repo.

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
decisions are in `final_report/02_the_data.md` and its Appendix C.

### BreastMNIST, ResNet-50 embeddings (`medmnist_breastmnist/`)

The second dataset (task 4): not gene expression. 780 breast ultrasound
images, malignant (210) vs benign or normal (570), from MedMNIST v2 (Yang et
al., 2023), 224 x 224 version. Each image becomes 2,048 ResNet-50 embedding
dimensions, so there are more features than images, as in SCAN-B.

- `breastmnist.py` downloads the images from Zenodo, checks the published
  MD5, and exposes them unchanged. It computes the embeddings once (ImageNet
  ResNet-50 without its last layer, on the GPU if there is one) and caches
  them. Computing needs the `embeddings` extra (`torch`, `torchvision`);
  reading cached embeddings does not.
- `featsel_view.py` defines `NAME`, `TARGET` (`label`) and `load_xy`: all 780
  images, MedMNIST's own split ignored.
- `demos/tutorial.ipynb` shows the images, labels and embeddings;
  `demos/selection.ipynb` compares selectors, simulates leakage and
  summarizes the full run.

Experiment config: `configs/experiment_breastmnist.yaml`.

### FGVC-Aircraft, ResNet-50 embeddings (`fgvc_aircraft/`)

A third dataset, unseen during the design: not medical, not genes. 1,200
aircraft photos from FGVC-Aircraft 2013b (Maji et al., 2013), Boeing 737
(800) against the Airbus A320 family (400). Each photo becomes 2,048
ResNet-50 features. The images are for non-commercial research only and
are not redistributed.

- `aircraft.py` downloads the 2.75 GB archive once and reads it in one
  pass without unpacking it. It crops each photo's copyright banner and
  embeds all 10,000 photos once (ImageNet ResNet-50, V1 weights, on the GPU
  if there is one; about 4 minutes on a GTX 1650). It also stores a 64-bit
  difference hash per photo, for finding near-duplicates.
- `featsel_view.py` defines `NAME`, `TARGET` (`family`) and `load_xy`: the
  two families from all three official splits.
- `demos/research.ipynb` studies the data, runs one selection step by step
  on the GPU, analyses the full run and states the conclusions.

Experiment config: `configs/aircraft70_gpu.yaml`, every selector on the GPU.

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
