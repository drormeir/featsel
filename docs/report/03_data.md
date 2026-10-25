# Data

Draft. Findings are recorded here as they are made, and rewritten as prose
later. Terms: a *patient* has one tumour; a *sample* is one sequencing run of
that tumour; a *gene* is one feature.

## SCAN-B (GSE96058)

The breast cancer data comes from the Sweden Cancerome Analysis Network -
Breast (SCAN-B). It was published with Brueffer et al. (2018) and deposited
in NCBI GEO as accession GSE96058.

- 3,273 patients, 3,409 samples, 30,865 genes.
- The smallest gene value equals log2(0.1). This fits a log2(FPKM + 0.1)
  transform, but the transform is not confirmed from the source.
- Samples were sequenced on two machines: 3,069 on a HiSeq 2000 and 340 on a
  NextSeq 500.
- The course's copies, `scanb_small` and `scanb_full`, hold exactly the 3,069
  HiSeq 2000 samples. Their gene values and labels match GSE96058.

The loader is `featsel_datasets/scanb_gse96058/`. It downloads the published
files and keeps them as a local cache.

**Decision: HiSeq 2000 samples only.** Two machines can give different gene
values for technical reasons alone. Mixing them would add a difference that
has nothing to do with the tumour. Using HiSeq only also keeps the same
patients as the course's copies. The NextSeq samples add 297 patients not in
the HiSeq set; they are left out.

## Repeat samples of one patient

**Finding.** The 3,069 HiSeq samples come from only 2,976 patients. 93
patients have two samples each.

**Source.** Brueffer et al. sequenced 136 tumours a second time to test the
reproducibility of their classifiers. GSE96058 publishes each repeat as its
own sample, with the suffix `repl` (for example `F2621` and `F2621repl`). 93
of the repeats are HiSeq samples. The course's copies kept them.

**They are not exact copies.** A patient's two samples have slightly different
gene values. Their correlation has a median of 0.970 and a minimum of 0.901.
Samples of two different patients correlate at a median of 0.925 and at most
0.951. In 87 of the 93 patients both samples carry the same PAM50 subtype.

**Impact.** When a patient's two samples fall on opposite sides of a
train/test split, the model is partly tested on a patient it has seen. Every
result computed on `scanb_small` or `scanb_full` has this leak. Its size is
not measured.

**Decision.** Keep one sample per patient: the first in file order. The
GSE96058 loader does this by default (`one_per_specimen=True`).

## Labels: measured and predicted

GSE96058 gives two kinds of label per patient.

- **Measured by pathologists:** ER, PgR, HER2, Ki67 status and grade.
- **Predicted from the same gene expression:** the classifiers of Brueffer et
  al. Single-gene classifiers (`sgc`) use one gene each. Multigene classifiers
  (`mgc`) are nearest shrunken centroid models; the ER one uses 459 genes.

PAM50 subtype is also computed from gene expression. The PAM50 algorithm
assigns it from 50 fixed genes.

**ER in the course's copies is a prediction.** It equals GSE96058's
`er prediction mgc` for all 3,069 samples, coded 1 = negative and
2 = positive. It disagrees with the pathologist on 235 of the 2,870 samples
that have a pathologist value. The pathologist value is missing for 199
samples.

**Consequence.** Predicting ER or PAM50 from gene expression partly re-learns
another classifier. A selector is expected to find the genes that classifier
used. Measured labels and survival do not have this circularity.

An earlier course report on this data (`references/local/`, unpublished)
treats ER and PAM50 as clinical facts. It found ER "somewhat trivial" to
predict, which is expected for a label computed from the same genes. It also
reduced PAM50 to two classes without saying how.

### Planned: reproducing the ER label

Decided 2026-10-17. ER is a target in this project, as in the earlier course
report. Unlike that report, this one states where the ER label comes from and
reproduces it.

Plan:

1. Rebuild Brueffer et al.'s ER multigene classifier. Supplementary Table 3
   gives its shrunken centroids per gene (459 nonzero for ER). The gene means
   and spreads it also needs come from the GSE81538 training cohort (405
   patients, with the pathologists' consensus labels).
2. Apply it to the GSE96058 samples. Compare its calls with
   `er prediction mgc` and with Supplementary Table 8.
3. Report the agreement, and any settings that had to be inferred, such as
   the offset and class priors of the nearest shrunken centroid method.

Sources are in `references/local/`: the paper and its Data Supplement.
GSE81538 is not downloaded yet.

### Planned: the source of PAM50, and recovering its genes

Decided 2026-10-17. The report discusses where the PAM50 label comes from,
and shows an experiment on whether feature selection recovers the genes that
define it.

What is known:

- PAM50 is the classifier of Parker et al. (2009). Each subtype has a fixed
  centroid over 50 genes. A sample gets the subtype of the nearest centroid,
  measured by Spearman rank correlation. Given the centroids, the subtype is
  a function of that sample's 50 gene values.
- The 50 genes were chosen from 161 candidate genes by ranking each gene
  alone with a t-test per subtype. A univariate filter such as ANOVA works
  the same way, so it has a built-in advantage here. Genes outside the 161
  were never candidates.
- The list is in Parker et al.'s Supplemental Table 2, with 5 control genes.
  Three symbols changed since: CDCA1 is NUF2, KNTC2 is NDC80, ORC6L is ORC6.
  `references/local/Parker_2009_PAM50/pam50_genes.csv` holds both forms.
- Brueffer et al. cite Parker et al. but neither the paper nor its Data
  Supplement says how GSE96058's `pam50 subtype` was computed.

Plan:

1. Find how SCAN-B assigned `pam50 subtype`: which implementation, and on
   which data. Look in other SCAN-B papers.
2. The experiment: which selection method recovers the 50 PAM50 genes most
   easily. The labels are not recomputed; only the gene subset is compared.
   Because the label is computed from these genes, the answer is known.
   Each method ranks all genes once per split. Walking down the ranking from
   `k = 50`, record how many PAM50 genes are found at each `k`. The result
   is a recall curve per method. The random control is the straight line
   `k / number of genes`. A strong method reaches the full list at a small
   `k`. See "Overlap with the published PAM50 gene list" in
   `docs/experiment_design.md`.

   Open, to solve later:
   - Scoring. The `k` that reaches all 50 is set by the single weakest gene,
     which may rank thousands deep for every method. Candidates: report the
     whole curve, plus the `k` at 90% and at 100%.
   - Methods without a full ranking. FDR, FWE, Lasso without `n_features`
     and Higher Criticism return a set, not a ranking. They give one point,
     not a curve.
   - Correlated substitutes. A method that picks a gene highly correlated
     with a PAM50 gene finds the same signal but scores no recall.
3. Run it on GSE96058, which holds all 50 genes; the retired `scanb_small`
   did not.

### Found: signal is widespread, and genes are correlated

Found 2026-10-24, in the smoke test (`configs/smoke_scanb.yaml`, one split).

**The facts.**

- The PAM50 label is computed from 50 genes. But each of them moves together
  with hundreds of other genes across patients. ESR1 moves with the
  estrogen-response genes; the proliferation genes move with the cell-cycle
  genes.
- The F-test scores each gene alone. A gene that moves with ESR1 differs
  between Basal and the rest as strongly as ESR1 does.
- So in every subtype-vs-rest task, thousands of genes carry signal: more
  than 10% of the 28,552.

**Consequences for Higher Criticism.** HC assumes two things that SCAN-B
does not meet.

- Rare signal: only a small fraction of genes carry signal. HC searches the
  smallest 10% of p-values (alpha0 = 0.1, as in both papers). On SCAN-B the
  HC score is still rising at that edge. HC2008 kept exactly 2,855 genes on
  every task, the edge itself. It therefore acted as ANOVA at k = 2,855, not
  as a method that picks its own count. These runs used the F-test; HC has
  used the U test since 2026-10-25 (below), and is not yet rerun.
- Independent genes: both papers derive the null behaviour of the sorted
  p-values, including the mean i/(N+1), from independent p-values.
  Correlated genes give correlated p-values.
- HC2004 kept one gene on every task. Its 1/sqrt(p) term lets the smallest
  p-value win alone.

The report states both unmet assumptions as the reason HC behaves this way
on SCAN-B. BreastMNIST is where to check whether HC behaves differently: what
fraction of its 2,048 embedding features carry signal is not yet checked.

Open: keep alpha0 = 0.1 and report the edge, or raise it so HC finds its
own peak. Raising it departs from both papers and must be stated.

**Consequence for recovering the PAM50 genes.** A selector can predict well
from genes that move with the 50, and still find few of the 50 themselves.
Low recovery does not mean poor selection. This is the "correlated
substitutes" point above, now explained.

### Decided: HC scores genes by the Mann-Whitney U test

Decided 2026-10-25.

**Why the U test fits the data.** The stored values are not measurements
in their own right. The GEO processing notes say the authors summed FPKM per
gene symbol, added 0.1 and took log2. FPKM is already a normalized number,
and the pseudocount and the log are a convenience; any other monotone
transform would have been equally valid. The PAM50 labels come without the
classifier's own scores. So what each task gives us, per gene, is a set of
values with a binary label, in which only the order of the values is
meaningful. The question we ask of a gene is how well that order separates
the two classes, which is its AUC, and how likely a separation this good is
by chance, with the class sizes n1 and n2 fixed by the dataset. That is
exactly the Mann-Whitney U test: U is the AUC times n1 n2, and its p-value
is the share of random label assignments that rank the classes at least
this well.

**Why the F-test does not.** The F-test compares class means, so its result
depends on the size of the values, and would change under another,
equally valid transform. Its p-values also assume that within each class a
gene's values follow a normal curve with the same spread in both classes.
The data is far from that: 52% of the values in the genes checked sit at
the floor, log2(0.1), meaning the gene was not detected, and 96.7% of genes
reject normality (`docs/experiment_design.md`). Within a class, a gene is
a heap of identical floor values plus a spread of detected ones, not a
normal curve. The F statistic still ranks genes sensibly, but its tiny
p-values, the ones HC reads, rest on that unmet assumption. The U test
needs neither the size of the values nor normality, and treats the floor
as what it is: one large group of tied values. It also needs exactly two
classes, which matches the project's binary-only tasks. HigherCriticismSelector
therefore takes its p-values from the two-sided U test. Ties get average
ranks; the p-value is the tie-corrected normal approximation, as in scipy,
with its log taken from the normal tail so nothing underflows.

**Limit.** The approximation is not exact in the extreme tail, where HC2004
reads its p-values. With ties, the exact permutation p-value depends on
each gene's own tie pattern, so it cannot be computed once for all genes.

**A better model, beyond this project's scope.** The data is better
described per gene by four numbers: n1 and n2, the class sizes given by the
dataset, and p1 and p2, the share of each class whose value sits at the
floor. This matches what we observe. In the first 200 genes, checked on
all 3,069 HiSeq samples, 52% of the values are at the floor, so the floor
is not an occasional tie but the main feature of the data. The model also separates the two ways a gene can differ between
classes: how often it is detected at all (p1 against p2), and how high it
is when detected (the order of the values above the floor). The U test
merges both into one AUC. Under the null, given the four numbers, the
labels split the floor values by a hypergeometric draw and the detected
values by a random order. That gives an exact null for each gene, and so
exact tail p-values, where the normal approximation is weakest. It needs a
separate computation per floor count, and is not built.

## Gene sets

- `scanb_full`: all 30,865 genes, including all 50 PAM50 genes. Across all
  3,409 GSE96058 samples, 2,236 genes have a single value and carry no
  information.
- `scanb_small`: 9,264 genes, chosen by the course. The selection rule is not
  recorded. It lacks 11 of the 50 PAM50 genes: ANLN, BAG1, CDC6, CXXC5,
  FOXA1, GPR160, KRT14, MIA, NUF2, TMEM45B and UBE2T.

## BreastMNIST (second dataset, task 4)

Terms here: a *sample* is one ultrasound image; a *feature* is one embedding
dimension.

**Source.** BreastMNIST is part of MedMNIST v2 (Yang et al., 2023), a
published benchmark collection. Its 780 breast ultrasound images come from
Al-Dhabyani et al. (2020). This project uses the 224 x 224 version
(MedMNIST+, Zenodo record 10519652, CC BY 4.0). The loader checks the file
against the published MD5.

- MedMNIST merged the source's *normal* (133) and *benign* (437) into one
  class against *malignant* (210): 570 vs 210.
- MedMNIST fixes a split of 546 / 78 / 156 images.

**Features.** Each image goes through an ImageNet ResNet-50 (torchvision,
`IMAGENET1K_V2` weights) with its last layer removed, giving 2,048 numbers.
The grayscale image is repeated over three channels and normalized as
ImageNet images are. The embeddings are computed once and cached; on this
machine, two runs on the GPU gave identical values.

- 2,048 features for 780 images: more features than samples, as in SCAN-B.
- Values are never negative; 42.8% are exactly zero (ReLU then average pool).
- Pairs of dimensions are weakly correlated: median |r| 0.056, and 0.17% of
  pairs above 0.5. Yet 12 principal components hold 50% of the variance, 150
  hold 90% and 462 hold 99%. The dependence is spread across many dimensions,
  not concentrated in strongly correlated pairs.
- 223 of 2,048 dimensions pass a univariate F test at p < 0.05 / 2048.

**Decision: all 780 images, MedMNIST's split ignored.** The runner draws its
own 100 stratified splits, as for SCAN-B. Results are therefore not directly
comparable with MedMNIST's published test-set scores (ResNet-18 trained end
to end: AUC 0.891 to 0.901).

**Trap.** Some images carry measurement marks ("+" calipers) drawn during the
scan. A model can learn such marks instead of the tissue, and the embeddings
include them. Not measured.

Code: `featsel_datasets/medmnist_breastmnist/`. Notebooks: `demos/tutorial.ipynb`
(the facts above) and `demos/selection.ipynb` (selector comparison, leakage
simulation, full run).

## Open

- How the course chose the 9,264 genes of `scanb_small`.
- How much the repeat samples inflate existing results.
- Which target is primary. The specification asks for "the clinical class"
  and does not name PAM50.
