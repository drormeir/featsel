# Data

Draft. Findings are recorded here as they are made, and rewritten as prose
later. Terms: a *patient* has one tumour; a *sample* is one sequencing run of
that tumour; a *gene* is one feature.

## SCAN-B (GSE96058)

The breast cancer data comes from the Sweden Cancerome Analysis Network -
Breast (SCAN-B). It was published with Brueffer et al. (2018) and deposited
in NCBI GEO as accession GSE96058.

- <!-- fact: scanb_patients -->3,273<!-- /fact --> patients, <!-- fact: scanb_samples -->3,409<!-- /fact --> samples, <!-- fact: scanb_genes -->30,865<!-- /fact --> genes.
- The smallest gene value equals log2(0.1). This fits a log2(FPKM + 0.1)
  transform, but the transform is not confirmed from the source.
- Samples were sequenced on two machines: <!-- fact: scanb_hiseq_samples -->3,069<!-- /fact --> on a HiSeq 2000 and
  <!-- fact: scanb_nextseq_samples -->340<!-- /fact --> on a NextSeq 500.
- The course's copies, `scanb_small` and `scanb_full`, hold exactly the
  <!-- fact: course_samples -->3,069<!-- /fact --> HiSeq 2000 samples. Their gene values and labels match GSE96058.

The loader is `featsel_datasets/scanb_gse96058/`. It downloads the published
files and keeps them as a local cache.

**Decision: HiSeq 2000 samples only.** Two machines can give different gene
values for technical reasons alone. Mixing them would add a difference that
has nothing to do with the tumour. Using HiSeq only also keeps the same
patients as the course's copies. The NextSeq samples add <!-- fact: scanb_nextseq_only_patients -->297<!-- /fact --> patients not
in the HiSeq set; they are left out.

## Repeat samples of one patient

**Finding.** The <!-- fact: scanb_hiseq_samples -->3,069<!-- /fact --> HiSeq samples come from only
<!-- fact: scanb_hiseq_patients -->2,976<!-- /fact --> patients. <!-- fact: scanb_repeat_patients -->93<!-- /fact --> patients have two samples each.

**Source.** Brueffer et al. sequenced 136 tumours a second time to test the
reproducibility of their classifiers. GSE96058 publishes each repeat as its
own sample, with the suffix `repl` (for example `F2621` and `F2621repl`). <!-- fact: scanb_repeat_patients -->93<!-- /fact -->
of the repeats are HiSeq samples. The course's copies kept them.

**They are not exact copies.** A patient's two samples have slightly different
gene values. Their correlation (Pearson r over all genes) has a median of
<!-- fact: scanb_repeat_r_median -->0.970<!-- /fact --> and a minimum of <!-- fact: scanb_repeat_r_min -->0.901<!-- /fact -->. The first samples of two different patients
correlate at a median of <!-- fact: scanb_between_r_median -->0.927<!-- /fact --> and at most <!-- fact: scanb_between_r_max -->0.968<!-- /fact -->. In <!-- fact: scanb_repeat_same_pam50 -->87<!-- /fact --> of the
<!-- fact: scanb_repeat_patients -->93<!-- /fact --> patients both samples carry the same PAM50 subtype.

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
`er prediction mgc` for <!-- fact: course_er_equals_mgc -->3,069<!-- /fact --> of its <!-- fact: course_samples -->3,069<!-- /fact --> samples, coded 1 = negative and
2 = positive. It disagrees with the pathologist on <!-- fact: scanb_er_disagree -->235<!-- /fact --> of the <!-- fact: scanb_er_known -->2,870<!-- /fact -->
HiSeq samples that have a pathologist value. The pathologist value is missing
for <!-- fact: scanb_er_missing -->199<!-- /fact --> samples.

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
  than 10% of the <!-- fact: scanb_loaded_genes -->28,552<!-- /fact -->.

**Consequences for Higher Criticism.** HC assumes two things that SCAN-B
does not meet.

- Rare signal: only a small fraction of genes carry signal. HC searches the
  smallest 10% of p-values (alpha0 = 0.1, as in both papers). On SCAN-B the
  HC score is still rising at that edge. In the selector comparison HC2008
  kept <!-- fact: scanb_hc2008_min -->2,855<!-- /fact --> to <!-- fact: scanb_hc2008_max -->2,856<!-- /fact --> genes on every task and split, the edge itself. It
  therefore acted as a top-k filter at the edge, not as a method that picks
  its own count. The first runs used the F-test and behaved the same.
- Independent genes: both papers derive the null behaviour of the sorted
  p-values, including the mean i/(N+1), from independent p-values.
  Correlated genes give correlated p-values.
- HC2004 kept <!-- fact: scanb_hc2004_min -->1<!-- /fact --> to <!-- fact: scanb_hc2004_max -->4<!-- /fact --> genes. Its 1/sqrt(p) term lets the smallest
  p-values win alone.

The report states both unmet assumptions as the reason HC behaves this way
on SCAN-B. BreastMNIST is where to check whether HC behaves differently: what
fraction of its 2,048 embedding features carry signal is not yet checked.

**Rerun with the U test, 2026-10-31.** The selector comparison
(`docs/comparison_scanb.md`, 5 splits) used the U test, and gave the counts
above: the change of test changed nothing.

**BreastMNIST, checked 2026-10-31: its signal is not rare either.** On each
of the 5 training halves, <!-- fact: bm_bh_share_min -->24%<!-- /fact --> to <!-- fact: bm_bh_share_max -->28%<!-- /fact --> of the <!-- fact: bm_features -->2,048<!-- /fact --> embedding features pass
the U test at a 5% false discovery rate (Benjamini-Hochberg), and
<!-- fact: bm_p05_share_min -->36%<!-- /fact --> to <!-- fact: bm_p05_share_max -->39%<!-- /fact --> have p < 0.05. HC searches only the smallest 10%. HC2008
again kept the edge, <!-- fact: breastmnist_hc2008_min -->204<!-- /fact --> to <!-- fact: breastmnist_hc2008_max -->204<!-- /fact --> features, on every split (`docs/comparison_breastmnist.md`).
So on both datasets the search limit, not the data, sets HC2008's count.

Decided 2026-10-31: alpha0 stays 0.1, as in both papers. A selector must
not assume what it is meant to find, and we treat the genes that define the
label as unknown. Raising alpha0 because we know SCAN-B's signal is
widespread would tune HC to the answer. The report states that on SCAN-B HC
reached the edge of its search range, and why.

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
The data is far from that: <!-- fact: scanb_floor_share_200 -->51%<!-- /fact --> of the values in the first 200 genes
sit at the floor, log2(0.1), meaning the gene was not detected, and
<!-- fact: scanb_non_normal_share -->98.1%<!-- /fact --> of genes reject normality (D'Agostino-Pearson test at p < 0.05,
median excess kurtosis <!-- fact: scanb_median_excess_kurtosis -->1.44<!-- /fact -->). Within a class, a gene is
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
all <!-- fact: scanb_hiseq_samples -->3,069<!-- /fact --> HiSeq samples, <!-- fact: scanb_floor_share_200 -->51%<!-- /fact --> of the values are at the floor, so the floor
is not an occasional tie but the main feature of the data. The model also separates the two ways a gene can differ between
classes: how often it is detected at all (p1 against p2), and how high it
is when detected (the order of the values above the floor). The U test
merges both into one AUC. Under the null, given the four numbers, the
labels split the floor values by a hypergeometric draw and the detected
values by a random order. That gives an exact null for each gene, and so
exact tail p-values, where the normal approximation is weakest. It needs a
separate computation per floor count, and is not built.

## Gene sets

- `scanb_full`: all <!-- fact: scanb_genes -->30,865<!-- /fact --> genes, including <!-- fact: scanb_pam50_present -->50<!-- /fact --> of the 50 PAM50 genes. Across
  all <!-- fact: scanb_samples -->3,409<!-- /fact --> GSE96058 samples, <!-- fact: scanb_single_value_genes -->2,236<!-- /fact --> genes have a single value and carry no
  information.
- `scanb_small`: <!-- fact: small_genes -->9,264<!-- /fact --> genes, chosen by the course. The selection rule is not
  recorded. It lacks <!-- fact: small_pam50_missing -->11<!-- /fact --> of the 50 PAM50 genes: <!-- fact: small_pam50_missing_list -->ANLN, BAG1, CDC6, CXXC5, FOXA1, GPR160, KRT14, MIA, NUF2, TMEM45B, UBE2T<!-- /fact -->.

## BreastMNIST (second dataset, task 4)

Terms here: a *sample* is one ultrasound image; a *feature* is one embedding
dimension.

**Source.** BreastMNIST is part of MedMNIST v2 (Yang et al., 2023), a
published benchmark collection. Its <!-- fact: bm_images -->780<!-- /fact --> breast ultrasound images come from
Al-Dhabyani et al. (2020). This project uses the 224 x 224 version
(MedMNIST+, Zenodo record 10519652, CC BY 4.0). The loader checks the file
against the published MD5.

- MedMNIST merged the source's *normal* (133) and *benign* (437) images,
  counts from Al-Dhabyani et al., into one class against *malignant*:
  <!-- fact: bm_benign_or_normal -->570<!-- /fact --> vs <!-- fact: bm_malignant -->210<!-- /fact -->.
- MedMNIST fixes a split of <!-- fact: bm_split_train -->546<!-- /fact --> / <!-- fact: bm_split_val -->78<!-- /fact --> / <!-- fact: bm_split_test -->156<!-- /fact --> images.

**Features.** Each image goes through an ImageNet ResNet-50 (He et al. 2016; torchvision,
`IMAGENET1K_V2` weights) with its last layer removed, giving 2,048 numbers.
The grayscale image is repeated over three channels and normalized as
ImageNet images are. The embeddings are computed once and cached; on this
machine, two runs on the GPU gave identical values.

- <!-- fact: bm_features -->2,048<!-- /fact --> features for <!-- fact: bm_images -->780<!-- /fact --> images: more features than samples, as in SCAN-B.
- Values are never negative; <!-- fact: bm_zero_share -->42.8%<!-- /fact --> are exactly zero (ReLU then average pool).
- Pairs of dimensions are weakly correlated: median |r| <!-- fact: bm_median_abs_r -->0.056<!-- /fact -->, and
  <!-- fact: bm_pairs_above_half -->0.17%<!-- /fact --> of pairs above 0.5. Yet <!-- fact: bm_pc_50 -->12<!-- /fact --> principal components hold 50% of the
  variance, <!-- fact: bm_pc_90 -->150<!-- /fact --> hold 90% and <!-- fact: bm_pc_99 -->462<!-- /fact --> hold 99%. The dependence is spread across many dimensions,
  not concentrated in strongly correlated pairs.
- <!-- fact: bm_f_bonferroni -->223<!-- /fact --> of <!-- fact: bm_features -->2,048<!-- /fact --> dimensions pass a univariate F test at p < 0.05 / 2048.

**Decision: all 780 images, MedMNIST's split ignored.** The runner draws its
own stratified splits, as for SCAN-B. Results are therefore not directly
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
