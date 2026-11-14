# 2. The data

[Back to contents](README.md) · Previous: [1. The problem](01_the_problem.md) · Next: [3. How to measure](03_how_to_measure.md)

The project uses two main datasets. The first, breast cancer gene
expression, comes with the specification. The second, ultrasound images,
answers task 4 with another domain. A third, aircraft photos, was added
after the design was fixed. It tests the pipeline on data it was not
built for (section 8.8).

## 2.1 SCAN-B breast cancer gene expression

**Source.** The Sweden Cancerome Analysis Network - Breast (SCAN-B)
sequenced the tumours of thousands of patients. The data was published
with Brueffer et al. (2018) in the public NCBI GEO archive, accession
GSE96058. Our loader downloads it once (592 MB) and caches it.

**Size.** The archive holds 3,273 patients, 3,409 samples and 30,865
genes. After the choices below, **2,976 patients and 28,541 genes**
remain.

| Choice | Why |
|---|---|
| Keep only HiSeq 2000 samples: 3,069 of 3,409. The other 340 are NextSeq 500. | Two machines can differ for technical reasons alone. The NextSeq samples add 297 patients not in the HiSeq set. They are left out. |
| Keep one sample per patient, the first in file order: 2,976 patients. | 93 of them have a HiSeq sample and its repeat. Split between the training set and the test set, they would test the model on a patient it has seen. |
| Drop 2,313 genes with a single value across the kept patients. | They carry no information. |
| Drop 11 flat genes: their range is under 1% of their largest absolute value. | Their largest change is 0.007 log2 units, about 0.5%. Standardized in float32, they become noise. On LumB vs rest, that noise made Lasso's solver about five times slower on the CPU. |

**The repeat samples.** Brueffer et al. sequenced 136 tumours twice, to
test their classifiers' reproducibility. GSE96058 lists each repeat as
its own sample, with the suffix `repl` (for example `F2621` and
`F2621repl`). 100 repeats are HiSeq samples. For 93 patients, both
samples are HiSeq.

The two samples of a patient are not exact copies. Their Pearson
correlation over all genes has median 0.970 and minimum 0.901. Two
different patients correlate at median 0.927 and at most 0.968. So a
repeat pair is usually, but not always, more alike than two patients. 87
of the 93 pairs share the PAM50 subtype.

**The course's copies.** The course provided `scanb_small` and
`scanb_full`. Both hold exactly the 3,069 HiSeq samples, with GSE96058's
PAM50 labels. We compared values only on 7 genes of `scanb_full`. They
matched to rounding (about 10⁻¹⁶). Both copies keep the repeat samples,
so every result on them carries the leak above. Its size was not
measured. `scanb_small` also keeps only 9,264 genes, by an unrecorded
rule. It lacks 11 of the 50 PAM50 genes: ANLN, BAG1, CDC6, CXXC5, FOXA1,
GPR160, KRT14, MIA, NUF2, TMEM45B and UBE2T. So we load GSE96058
directly.

### What the numbers are

Each value is log2(FPKM + 0.1). The GEO processing notes say the authors
summed FPKM per gene symbol, added 0.1 and took log2. The data agrees: its
smallest value is log2(0.1), to every stored digit. FPKM is a normalized
count of how much of a gene's RNA was read. The +0.1 and the log are a
convenience. Any other increasing transform would be equally valid. Two
facts follow.

- **Only the order of the values is meaningful.** Their size depends on an
  arbitrary transform.
- **Many values sit at a floor.** An undetected gene gets log2(0.1). In
  the first 200 genes, over the 3,069 HiSeq samples, 51% of values are at
  the floor. Over the 2,976 kept patients, 98.1% of genes fail a normality
  test (D'Agostino-Pearson, p < 0.05). The median excess kurtosis is
  1.44. Within a class, a gene is a heap at the floor plus a spread of
  detected values, not a normal curve.

These facts decided the test that scores genes
([chapter 4](04_the_methods.md), section 4.3).

### The labels: measured and computed

GSE96058 gives two kinds of label per patient.

- **Measured by pathologists:** estrogen receptor (ER) status, PgR, HER2,
  Ki67 and grade.
- **Computed from the same gene expression:** the PAM50 subtype, and the
  outputs of Brueffer et al.'s classifiers, in fields such as
  `er prediction mgc`. Single-gene classifiers (`sgc`) use one gene each.
  Multigene classifiers (`mgc`) are nearest shrunken centroid models. The
  ER one uses 459 genes.

**PAM50** (Parker et al. 2009), our target, assigns each tumour one of
five subtypes: Basal, Her2, LumA, LumB or Normal. Each subtype has a
fixed reference profile, a centroid, over 50 genes. A tumour gets the
subtype of the nearest centroid by Spearman rank correlation. So the
label is a function of 50 genes that are in our data.

**How the 50 genes were chosen.** Parker et al. ranked 161 candidate
genes one at a time, by a t-test per subtype, and kept the best. Genes
outside the 161 were never candidates. A univariate filter such as
Mann-Whitney also ranks genes one at a time. So it has a built-in
advantage in finding them. The 50 genes are shown in Parker et al.'s
Appendix Figure A2 and named in their Supplemental Table 2, beside five
reference genes. Three symbols have changed since: CDCA1 is now NUF2,
KNTC2 is NDC80, and ORC6L is ORC6.

**What is not known.** Brueffer et al. cite Parker et al. But their paper
and its supplement do not describe GSE96058's `pam50 subtype` field. The
GEO processing notes say a little more. The authors used an
implementation of Parker's method. Before subtyping, their own R scripts
centred each tumour's PAM50 genes to a fixed reference set, chosen to
match Parker's cohort. The notes name neither the implementation nor the
reference samples. The code is not published.

A computed label has two consequences.

1. **Predicting PAM50 partly re-learns a published classifier.** Scores
   may be higher than on a measured clinical outcome.
2. **We know the right answer.** We can check whether a method finds the
   50 genes ([chapter 7](07_results_quality.md), section 7.6).

**The ER label in the course's copy is also computed, not measured.** It
equals `er prediction mgc` on all 3,069 samples, coded 1 = negative and
2 = positive. It disagrees with the pathologist on 235 of the 2,870
samples that have a pathologist value. The other 199 lack one. A label
computed from the genes is easy to predict from them. Measured labels and
survival lack this circularity.

### Five binary tasks

The project selects features for binary classifiers only. So PAM50's five
classes become five tasks, each subtype against the other four. Each task
has its own selected genes and scores:

| Task | Positive patients | Rest |
|---|---|---|
| Basal vs rest | 308 | 2,668 |
| Her2 vs rest | 297 | 2,679 |
| LumA vs rest | 1,502 | 1,474 |
| LumB vs rest | 669 | 2,307 |
| Normal vs rest | 200 | 2,776 |

Four of the five tasks are imbalanced. On Normal vs rest, a model that
always answers "rest" is right 93% of the time. So plain accuracy is not
the main score ([chapter 3](03_how_to_measure.md)). A single five-class
task was rejected: its classes are too imbalanced to report as one number.

### Signal is everywhere, and genes move together

Each of the 50 PAM50 genes rises and falls with hundreds of others across
patients. ESR1 moves with the estrogen-response genes. The proliferation
genes move with the cell-cycle genes. The U test scores each gene alone.
So a gene that moves with ESR1 separates Basal from the rest about as
strongly as ESR1 does.

As a result, more than 10% of all genes carry signal in every task. On
`scanb_small`, with the F-test, 4,792 to 6,964 of the 9,259 usable genes
per task had a p-value below 1/9,259. A one-split test on GSE96058 showed
the same. This explains three later results:

- Higher Criticism stops at the edge of its search (7.5).
- Few PAM50 genes are recovered (7.6).
- Random genes catch up fast (7.2).

Loader: [`featsel_datasets/scanb_gse96058/`](../featsel_datasets/scanb_gse96058/).
Tutorial: [`featsel_datasets/scanb_gse96058/demos/tutorial.ipynb`](../featsel_datasets/scanb_gse96058/demos/tutorial.ipynb).
Data exploration: [`notebooks/01_data_exploration.ipynb`](../notebooks/01_data_exploration.ipynb).

## 2.2 BreastMNIST image embeddings (task 4)

**Why this dataset.** Task 4 asks for a second dataset, not necessarily
gene expression. Like SCAN-B, BreastMNIST has more features than samples.
It comes from another domain. It is public and published as a benchmark.
We rejected the NIPS 2003 challenge sets: Gisette has fewer features than
samples, Madelon only 500 features, and Arcene only 200 samples.

**Source.** BreastMNIST is part of the MedMNIST v2 benchmark collection
(Yang et al. 2023). It holds 780 breast ultrasound images from
Al-Dhabyani et al. (2020). We use the 224 × 224 version (MedMNIST+,
Zenodo record 10519652, licence CC BY 4.0). The loader checks the
download against its published MD5 checksum. The source has three
classes: normal (133 images), benign (437) and malignant (210). MedMNIST
merged normal and benign. So the task is malignant (210) against benign
or normal (570). MedMNIST also fixes a split of 546 training, 78
validation and 156 test images.

**Features.** Each image goes through ResNet-50 (He et al. 2016) with its
last layer removed. ResNet-50 is a convolutional network trained on the
ImageNet photo collection. We use torchvision's `IMAGENET1K_V2` weights.
The grayscale image is repeated over three colour channels and normalized
as for ImageNet. The result, an embedding, is 2,048 numbers per image.
This is a common form of transfer learning: a network trained on one task
describes images for another. The embeddings are computed once on the GPU
and cached. Two GPU runs gave identical values.

| Fact | Value |
|---|---|
| Values exactly zero | 42.8% (ReLU, then average pooling) |
| Negative values | none |
| Features passing the U test at 5% false discovery rate | 24% to 28% |
| Features with U-test p < 0.05 | 36% to 39% |
| Features passing the F-test at p < 0.05 / 2,048 (Bonferroni) | 223 |
| Median absolute correlation between two features | 0.056 |
| Pairs of features with correlation above 0.5 | 0.17% |
| Principal components for 50% / 90% / 99% of the variance | 12 / 150 / 462 |

The U-test ranges are over the training sets of the comparison's first 5
splits (chapter 7).

As in SCAN-B, the signal is widespread: about a quarter of the features
separate the classes. Unlike SCAN-B, pairs of features are only weakly
correlated. Yet 150 principal components hold 90% of the variance of
2,048 features. So the features do depend on each other. The dependence
is spread thinly across many features, not concentrated in strongly
correlated pairs.

We use all 780 images and draw our own random splits, as for SCAN-B. We
ignore MedMNIST's fixed split. So our scores are not directly comparable
with its published test-set results, where ResNet-18 trained end to end
reached AUC 0.891 to 0.901.

**Defects in the images.** Some images carry "+" calipers, measurement
marks drawn during the scan. The embeddings include them, so a model can
learn them instead of the tissue. A later check of the source images
found worse defects (Pawłowska et al. 2023, a letter). BreastMNIST uses
all 780 source images (Yang et al. 2023), so all of these are in our data.

| Defect | Extent | Effect |
|---|---|---|
| "+" calipers | some images | not measured |
| Copies: the same scan, shifted, scaled or cropped | The letter's text gives 155 copies, 19% of the set. Its own list holds 153 groups of 346 images, so 193 images repeat another. Some copies carry a different label: in the letter's numbering, image 42 is benign and its copy 488 malignant. | compared in chapter 7 |
| The axilla rather than the breast | 70 images, more than 8% | not measured |
| A biopsy needle | at least 7 images | not measured |

The copies matter most. Like SCAN-B's repeat samples, a copy can fall in
the training set and its twin in the test set. The model is then partly
tested on an image it has seen.

**Finding the copies in BreastMNIST.** The letter names images by their
original file names, which BreastMNIST does not keep. So the loader finds
the copies among BreastMNIST's own images, by the letter's method:
geometric registration. It finds 150 groups of 339 images, against the
letter's 153 groups and 346 images. Even MedMNIST's fixed split separates
some copies from their twins. The method, its checks and examples are in
a report beside the loader:
[`copies_report.md`](../featsel_datasets/medmnist_breastmnist/copies_report.md).

The run without copies keeps one image per group and drops the 6 groups
with conflicting labels. 585 images remain: 180 malignant and 405 benign
or normal. Chapter 7 compares the results with and without the copies.

Loader: [`featsel_datasets/medmnist_breastmnist/`](../featsel_datasets/medmnist_breastmnist/).
Notebooks: [`demos/tutorial.ipynb`](../featsel_datasets/medmnist_breastmnist/demos/tutorial.ipynb),
[`demos/selection.ipynb`](../featsel_datasets/medmnist_breastmnist/demos/selection.ipynb).

## 2.3 Summary

| | SCAN-B | BreastMNIST | FGVC-Aircraft (section 8.8) |
|---|---|---|---|
| Domain | gene expression (RNA sequencing) | ultrasound images, ResNet-50 embeddings | aircraft photos, ResNet-50 embeddings |
| Samples | 2,976 patients | 780 images | 1,200 photos |
| Features | 28,541 genes | 2,048 embedding dimensions | 2,048 embedding dimensions |
| Tasks | 5 (each PAM50 subtype vs rest) | 1 (malignant vs rest) | 1 (Boeing 737 vs Airbus A320 family) |
| Label | computed from 50 known genes | assigned by the image source | the aircraft model, from the dataset |
| Signal | widespread, strongly correlated genes | widespread, weakly correlated features | widespread, weakly correlated features |

Plans and open questions about the data are in
[Appendix C](appendix_c_data_details.md).

Source: [`results/data_facts.csv`](../results/data_facts.csv).
