# 2. The data

[Back to contents](README.md) · Previous: [1. The problem](01_the_problem.md) · Next: [3. How to measure](03_how_to_measure.md)

The project uses two datasets. The first is the one the specification
provides: breast cancer gene expression. The second, required by task 4,
comes from a different domain: ultrasound images. A third, aircraft
photos, was added after the design was fixed, to test the pipeline on
data it was not built for. Section 8.8 describes it with its results.

## 2.1 SCAN-B breast cancer gene expression

**Source.** The Sweden Cancerome Analysis Network - Breast (SCAN-B)
sequenced the tumours of thousands of patients. The data was published
with Brueffer et al. (2018). It is in the public NCBI GEO archive,
accession GSE96058. Our loader downloads it once (592 MB) and caches it.

**Size.** The archive holds 3,273 patients, 3,409 samples and 30,865
genes. After the choices below, the project uses **2,976 patients and
28,552 genes**.

| Choice | Why |
|---|---|
| Keep only samples sequenced on the HiSeq 2000 machine (3,069 of 3,409). The other 340 were sequenced on a NextSeq 500. | Two machines can give different values for technical reasons alone. Mixing them would add a difference that has nothing to do with the tumour. The NextSeq samples add 297 patients not in the HiSeq set. They are left out. |
| Keep one sample per patient (2,976 patients), the first in file order. | 93 of these patients have two HiSeq samples: a sample and its repeat. If a patient's two samples fall on both sides of a train/test split, the model is tested on a patient it has already seen. |
| Drop genes with a single value across the kept patients (2,313 genes). | They carry no information. |

**The repeat samples.** Brueffer et al. sequenced 136 tumours a second
time, to test how reproducible their classifiers are. GSE96058 publishes
each repeat as its own sample, with the suffix `repl` (for example `F2621`
and `F2621repl`). 100 of the 136 repeats are HiSeq samples. For 93
patients, both the sample and its repeat are HiSeq samples.

The two samples of one patient are not exact copies. Their Pearson
correlation over all genes has a median of 0.970 and a minimum of 0.901. Two
different patients correlate at a median of 0.927, and at most 0.968. So a
repeat pair is usually, but not always, more alike than two different
patients. In 87 of the 93 patients both samples carry the same PAM50
subtype.

**The course's copies.** The course provided two copies of this data,
`scanb_small` and `scanb_full`. Both hold exactly the 3,069 HiSeq samples,
with the same PAM50 labels as GSE96058. Values were compared on 7 genes of
`scanb_full` only. They matched to rounding (about 10⁻¹⁶). Both kept the repeat
samples. So every result computed on them carries the leak above. Its
size was not measured. `scanb_small` also keeps only 9,264 genes, by an unrecorded rule.
It lacks 11 of the 50 PAM50 genes: ANLN, BAG1, CDC6, CXXC5, FOXA1, GPR160,
KRT14, MIA, NUF2, TMEM45B and UBE2T. We therefore load GSE96058 directly
and do not use the course copies.

### What the numbers are

Each gene's value is log2(FPKM + 0.1). The GEO processing notes say the
authors summed FPKM per gene symbol, added 0.1 and took log2. The data
agrees: its smallest value is log2(0.1), to every digit the file stores.
FPKM is a normalized count of how much of the gene's RNA was read. The
+0.1 and the log are a convenience chosen by the authors. Any other
increasing transform would be equally valid. Two facts follow:

- **Only the order of the values is meaningful.** Their size depends on an
  arbitrary choice of transform.
- **Many values sit at a floor.** A gene that was not detected gets
  log2(0.1). We checked the first 200 genes on all 3,069 HiSeq samples:
  51% of their values are at this floor. Over the 2,976 patients we keep,
  98.1% of genes fail a test of normality (D'Agostino-Pearson at
  p < 0.05). Their median excess kurtosis is 1.44. Within a class, a gene
  is a heap of identical floor values plus a spread of detected ones. It is
  not a normal curve.

These facts decided which statistical test we use to score genes
([chapter 4](04_the_methods.md), section 4.3).

### The labels: measured and computed

GSE96058 gives two kinds of label for each patient.

- **Measured by pathologists:** estrogen receptor (ER) status, PgR, HER2,
  Ki67 and grade.
- **Computed from the same gene expression:** the classifiers of Brueffer
  et al. Single-gene classifiers (`sgc`) use one gene each. Multigene
  classifiers (`mgc`) are nearest shrunken centroid models. The ER one
  uses 459 genes. Their outputs are stored as fields such as
  `er prediction mgc`. The PAM50 subtype is also computed.

**PAM50**, our target, is computed (Parker et al. 2009). It assigns each
tumour to one of five subtypes: Basal, Her2, LumA, LumB or Normal. Each
subtype has a fixed reference profile, a centroid, over 50 genes. A tumour
gets the subtype of the nearest centroid, measured by Spearman rank
correlation. So the label is a function of 50 genes that are in our data.

**How the 50 genes were chosen.** Parker et al. started from 161
candidate genes. They ranked each gene alone, with a t-test per subtype,
and kept the best. A univariate filter such as Mann-Whitney works the same
way. So it has a built-in advantage in finding them. Genes outside the 161
were never candidates. Parker et al. show the 50 genes in their Appendix
Figure A2. Their Supplemental Table 2 also names them, next to five
reference genes. Three symbols have changed since: CDCA1 is now NUF2,
KNTC2 is NDC80, and ORC6L is ORC6.

**What is not known.** Brueffer et al. cite Parker et al. But neither their
paper nor its supplement describes GSE96058's `pam50 subtype` field. The
GEO processing notes say a little more. The authors used an
implementation of Parker's method. Before subtyping, they centred each
tumour's PAM50 genes to a fixed reference set, chosen to match Parker's
cohort, with their own R scripts. The notes do not name the
implementation or the reference samples, and the code is not published.

This has two consequences.

1. **Predicting PAM50 partly re-learns a published classifier.** Scores on
   it may be higher than on a measured clinical outcome.
2. **We know the right answer.** We can check whether a method finds the
   50 genes ([chapter 7](07_results_quality.md), section 7.6).

**The ER label in the course's copy is also computed, not measured.** It
equals `er prediction mgc` on all 3,069 samples. It is coded 1 = negative
and 2 = positive. It disagrees with the pathologist on 235 of the 2,870
samples that have a pathologist value. The pathologist value is missing
for 199 samples. A label computed from the genes is easy to predict from
the same genes. Measured labels and survival do not have this circularity.

### Five binary tasks

The project selects features for binary classifiers only. PAM50 has five
classes. So it becomes five tasks, each subtype against the other four:

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

A five-class task was rejected. The classes are too imbalanced to report
as one number.

### Signal is everywhere, and genes move together

PAM50 uses 50 genes. But each of them rises and falls together with
hundreds of others across patients. For example, ESR1 moves with the
estrogen-response genes, and the proliferation genes move with the
cell-cycle genes. The U test scores each gene alone. So a gene that moves
with ESR1 separates Basal from the rest about as strongly as ESR1 does.
As a result, in every task thousands of genes carry signal: more than 10%
of all genes. On `scanb_small`, with the F-test, 4,792 to 6,964 of the
9,259 usable genes per task had a p-value below 1/9,259. A one-split test
on GSE96058 showed the same. This explains three later results. Higher Criticism stops at the edge of its
search (7.5). Few PAM50 genes are recovered (7.6). Random genes catch up
fast (7.2).

Loader: [`featsel_datasets/scanb_gse96058/`](../featsel_datasets/scanb_gse96058/).
Tutorial: [`featsel_datasets/scanb_gse96058/demos/tutorial.ipynb`](../featsel_datasets/scanb_gse96058/demos/tutorial.ipynb).
Data exploration: [`notebooks/01_data_exploration.ipynb`](../notebooks/01_data_exploration.ipynb).

## 2.2 BreastMNIST image embeddings (task 4)

**Why this dataset.** Task 4 asks for a second dataset, not necessarily
gene expression. We chose one with more features than samples, as in
SCAN-B. It comes from a different domain. It is public and published as a
benchmark. The NIPS 2003 challenge sets were
rejected. Gisette has fewer features than samples, Madelon only 500
features, and Arcene only 200 samples.

**Source.** BreastMNIST is part of MedMNIST v2 (Yang et al. 2023), a
published benchmark collection. It holds 780 breast ultrasound images
from Al-Dhabyani et al. (2020). We use the 224 × 224 version (MedMNIST+,
Zenodo record 10519652, licence CC BY 4.0). The loader checks the
downloaded file against its published MD5 checksum.

- The source has three classes: normal (133 images), benign (437) and
  malignant (210). MedMNIST merged normal and benign into one class. So the
  task is malignant (210) against benign or normal (570).
- MedMNIST fixes a split of 546 training, 78 validation and 156 test
  images.

**Features.** Each image goes through ResNet-50 (He et al. 2016) with its
last layer removed. ResNet-50 is a convolutional network trained on the
ImageNet photo collection. We use torchvision's `IMAGENET1K_V2` weights. The
grayscale image is repeated over three colour channels and normalized as
ImageNet images are. The result is 2,048 numbers per image, called an
embedding. This is a common form of transfer learning: a network trained
on one task describes images for another. The embeddings are computed
once on the GPU and cached. Two runs on the GPU gave identical values.

| Fact | Value |
|---|---|
| Images (samples) | 780 |
| Features | 2,048 |
| Task | malignant (210) vs benign or normal (570) |
| Values exactly zero | 42.8% (ReLU, then average pooling) |
| Negative values | none |
| Features passing the U test at 5% false discovery rate | 24% to 28% |
| Features with U-test p < 0.05 | 36% to 39% |
| Features passing the F-test at p < 0.05 / 2,048 (Bonferroni) | 223 |
| Median absolute correlation between two features | 0.056 |
| Pairs of features with correlation above 0.5 | 0.17% |
| Principal components for 50% / 90% / 99% of the variance | 12 / 150 / 462 |

The two share ranges are over the training halves of the comparison's
first 5 splits (chapter 7).

As in SCAN-B, the signal is widespread: about a quarter of the features
separate the classes. Unlike SCAN-B, pairs of features are only weakly
correlated. Yet 150 principal components are needed to hold 90% of the
variance. The dependence is spread thinly across many features. It is not
concentrated in strongly correlated pairs.

We use all 780 images and draw our own random splits, as for SCAN-B.
MedMNIST's fixed split is ignored. So our scores are not directly
comparable with its published test-set results. There, ResNet-18 trained
end to end reached AUC 0.891 to 0.901.

**Traps we did not measure.** Some images carry measurement marks: "+"
calipers drawn during the scan. A model can learn such marks instead of
the tissue, and the embeddings include them.

A later check of the source images found worse defects (Pawłowska et al.
2023). 155 of the 780 images, 19%, are copies of other images in the
set. Some copies carry a different label. For example, in the letter's
numbering, image 42 is benign and its copy 488 is malignant. 70 images,
more than 8%, show the axilla rather than the breast. At least 7 show a
biopsy needle. BreastMNIST uses all 780 images (Yang et al. 2023). So
these defects are in our data.

The copies matter most. Like SCAN-B's repeat samples, a copy can fall in
the training part while its twin is in the test part. The model is then
partly tested on an image it has seen. We found this after the
experiments. So every BreastMNIST result in this report carries the
leak. Its size is not measured.

Loader: [`featsel_datasets/medmnist_breastmnist/`](../featsel_datasets/medmnist_breastmnist/).
Notebooks: [`demos/tutorial.ipynb`](../featsel_datasets/medmnist_breastmnist/demos/tutorial.ipynb),
[`demos/selection.ipynb`](../featsel_datasets/medmnist_breastmnist/demos/selection.ipynb).

## 2.3 Summary

| | SCAN-B | BreastMNIST |
|---|---|---|
| Domain | gene expression (RNA sequencing) | ultrasound images, ResNet-50 embeddings |
| Samples | 2,976 patients | 780 images |
| Features | 28,552 genes | 2,048 embedding dimensions |
| Tasks | 5 (each PAM50 subtype vs rest) | 1 (malignant vs rest) |
| Label | computed from 50 known genes | assigned by the image source |
| Signal | widespread, strongly correlated genes | widespread, weakly correlated features |

Plans and open questions about the data are in
[Appendix C](appendix_c_data_details.md).

Source: [`results/data_facts.csv`](../results/data_facts.csv).
