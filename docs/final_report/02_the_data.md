# 2. The data

[Back to contents](README.md) · Previous: [1. The problem](01_the_problem.md) · Next: [3. How to measure](03_how_to_measure.md)

The project uses two datasets. The first is the one the specification
provides: breast cancer gene expression. The second, required by task 4,
comes from a different domain: ultrasound images.

## 2.1 SCAN-B breast cancer gene expression

**Source.** The Sweden Cancerome Analysis Network - Breast (SCAN-B)
sequenced the tumours of thousands of patients. The data was published
with Brueffer et al. (2018) in the public NCBI GEO archive, accession
GSE96058. Our loader downloads it once (592 MB) and caches it.

**Size.** The archive holds 3,273 patients, 3,409 samples and 30,865
genes. After the choices below, the project uses **2,976 patients and
28,552 genes**.

| Choice | Why |
|---|---|
| Keep only samples sequenced on the HiSeq 2000 machine (3,069 of 3,409). | Two machines can give different values for technical reasons alone. Mixing them would add a difference that has nothing to do with the tumour. |
| Keep one sample per patient (2,976 patients). | 93 patients were sequenced twice, to test reproducibility. If a patient's two samples fall on both sides of a train/test split, the model is tested on a patient it has already seen. |
| Drop genes with a single value across the kept patients (2,313 genes). | They carry no information. |

The course's own copy of the data kept the repeat samples. Every result
computed on it carries that leak. We measured that the two samples of one
patient are very similar (median correlation 0.970), more similar than
any two different patients (median 0.927).

### What the numbers are

Each gene's value is log2(FPKM + 0.1). FPKM is a normalized count of how
much of the gene's RNA was read. The +0.1 and the log are a convenience
chosen by the authors; any other increasing transform would be equally
valid. Two facts follow:

- **Only the order of the values is meaningful.** Their size depends on an
  arbitrary choice of transform.
- **Many values sit at a floor.** A gene that was not detected gets
  log2(0.1). In the first 200 genes, 51% of all values are at this floor.
  98.1% of genes fail a test of normality.

These facts decided which statistical test we use to score genes
([chapter 4](04_the_methods.md), section 4.3).

### The labels: measured and computed

GSE96058 gives two kinds of label for each patient.

- **Measured by pathologists:** estrogen receptor (ER) status, PgR, HER2,
  Ki67 and grade.
- **Computed from the same gene expression:** classifier outputs, such as
  `er prediction mgc` and the PAM50 subtype.

**PAM50**, our target, is computed (Parker et al. 2009). It assigns each
tumour to one of five subtypes, Basal, Her2, LumA, LumB or Normal, by
comparing 50 fixed genes with five reference profiles. So the label is a
function of 50 genes that are in our data. This has two consequences.

1. **Predicting PAM50 partly re-learns a published classifier.** Scores on
   it may be higher than on a measured clinical outcome.
2. **We know the right answer.** We can check whether a method finds the
   50 genes ([chapter 7](07_results_quality.md), section 7.5).

The ER label in the course's copy is also computed, not measured. It
equals `er prediction mgc` on all 3,069 samples, and it disagrees with the
pathologist on 235 of the 2,870 samples that have a pathologist value. An
earlier course report on this data treated both labels as clinical facts.

### Five binary tasks

The project selects features for binary classifiers only (decided
2026-10-18). PAM50 has five classes, so it becomes five tasks, each subtype
against the other four:

| Task | Positive patients | Rest |
|---|---|---|
| Basal vs rest | 308 | 2,668 |
| Her2 vs rest | 297 | 2,679 |
| LumA vs rest | 1,502 | 1,474 |
| LumB vs rest | 669 | 2,307 |
| Normal vs rest | 200 | 2,776 |

Four of the five tasks are imbalanced. On Normal vs rest, a model that
always answers "rest" is right 93% of the time. This is why we do not use
plain accuracy as the main score ([chapter 3](03_how_to_measure.md)).

A five-class task was rejected: the classes are too imbalanced to report
as one number.

### Signal is everywhere, and genes move together

PAM50 uses 50 genes. But each of them rises and falls together with
hundreds of others across patients. For example, ESR1 moves with the
estrogen-response genes, and the proliferation genes move with the
cell-cycle genes. As a result, in every task thousands of genes carry
signal: more than 10% of all genes. This fact explains several results
later in the report.

Loader: [`featsel_datasets/scanb_gse96058/`](../../featsel_datasets/scanb_gse96058/).
Tutorial: [`featsel_datasets/scanb_gse96058/demos/tutorial.ipynb`](../../featsel_datasets/scanb_gse96058/demos/tutorial.ipynb).
Data exploration: [`notebooks/01_data_exploration.ipynb`](../../notebooks/01_data_exploration.ipynb).

## 2.2 BreastMNIST image embeddings (task 4)

**Why this dataset.** Task 4 asks for a second dataset, not necessarily
gene expression. We chose one with more features than samples, as in
SCAN-B, from a different domain, public and published as a benchmark
(decided 2026-10-17).

**Source.** BreastMNIST is part of MedMNIST v2 (Yang et al. 2023). It
holds 780 breast ultrasound images (Al-Dhabyani et al. 2020): 210
malignant against 570 benign or normal. We use the 224 × 224 version.

**Features.** Each image goes through ResNet-50, a convolutional network
trained on the ImageNet photo collection, with its last layer removed. The
result is 2,048 numbers per image, called an embedding. This is a common
form of transfer learning: a network trained on one task describes images
for another. The embeddings are computed once on the GPU and cached.

| Fact | Value |
|---|---|
| Images (samples) | 780 |
| Features | 2,048 |
| Task | malignant (210) vs benign or normal (570) |
| Values exactly zero | 42.8% |
| Features passing the U test at 5% false discovery rate | 24% to 28% |

As in SCAN-B, the signal is widespread: about a quarter of the features
separate the classes. Unlike SCAN-B, pairs of features are only weakly
correlated (median |r| 0.056), but 150 principal components are needed to
hold 90% of the variance.

We use all 780 images and draw our own random splits, as for SCAN-B.
MedMNIST's fixed split is ignored, so our scores are not directly
comparable with its published results.

**A trap we did not measure.** Some images carry measurement marks drawn
during the scan. A model can learn such marks instead of the tissue.

Loader: [`featsel_datasets/medmnist_breastmnist/`](../../featsel_datasets/medmnist_breastmnist/).
Notebooks: [`demos/tutorial.ipynb`](../../featsel_datasets/medmnist_breastmnist/demos/tutorial.ipynb),
[`demos/selection.ipynb`](../../featsel_datasets/medmnist_breastmnist/demos/selection.ipynb).

## 2.3 Summary

| | SCAN-B | BreastMNIST |
|---|---|---|
| Domain | gene expression (RNA sequencing) | ultrasound images, ResNet-50 embeddings |
| Samples | 2,976 patients | 780 images |
| Features | 28,552 genes | 2,048 embedding dimensions |
| Tasks | 5 (each PAM50 subtype vs rest) | 1 (malignant vs rest) |
| Label | computed from 50 known genes | assigned by the image source |
| Signal | widespread, strongly correlated genes | widespread, weakly correlated features |

Sources: [`docs/report/03_data.md`](../report/03_data.md),
[`results/data_facts.csv`](../../results/data_facts.csv).
