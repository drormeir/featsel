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
3. Run it on a gene set that holds all 50 genes: GSE96058 or `scanb_full`,
   not `scanb_small`.

## Gene sets

- `scanb_full`: all 30,865 genes, including all 50 PAM50 genes. Across all
  3,409 GSE96058 samples, 2,236 genes have a single value and carry no
  information.
- `scanb_small`: 9,264 genes, chosen by the course. The selection rule is not
  recorded. It lacks 11 of the 50 PAM50 genes: ANLN, BAG1, CDC6, CXXC5,
  FOXA1, GPR160, KRT14, MIA, NUF2, TMEM45B and UBE2T.

## Open

- How the course chose the 9,264 genes of `scanb_small`.
- How much the repeat samples inflate existing results.
- Which target is primary. The specification asks for "the clinical class"
  and does not name PAM50.
