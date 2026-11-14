# Appendix C. Data: plans, open questions and extra facts

[Back to contents](README.md)

Chapter 2 describes the data. This appendix keeps the plans that were made
about it, what is still open, and facts too detailed for the chapter.

## C.1 Reproducing the ER label (planned, not done)

ER was planned as a target in this project. No experiment in this report
uses it. Chapter 2 states where the ER label comes from. This section
gives the plan to reproduce the published classifier behind it.

1. Rebuild Brueffer et al.'s ER multigene classifier. Their Supplementary
   Table 3 gives its shrunken centroids per gene. For ER, 459 genes have a
   non-zero centroid.
   The classifier also needs gene means and spreads. These come from the
   GSE81538 training cohort: 405 patients, with the pathologists'
   consensus labels.
2. Apply it to the GSE96058 samples. Compare its calls with
   `er prediction mgc` and with their Supplementary Table 8.
3. Report the agreement. Also report any settings that had to be
   inferred, such as the offset and the class priors of the nearest
   shrunken centroid method.

Two risks were known in advance. First, GSE81538 was mapped to an older
version of the human genome (hg19) than GSE96058 (hg38). So a few gene
names may not match. Second, the 459 genes were chosen automatically from
the 5,000 most variable genes of the 405 training patients. A rebuild from
scratch with another shrinkage threshold would give a different list.

## C.2 The source of PAM50, and recovering its genes

Chapter 2 discusses where the PAM50 label comes from. Section 7.6 tests
whether feature selection recovers the genes that define it.

**Still open.** How SCAN-B assigned `pam50 subtype`. The GEO notes say
only that an implementation of Parker's method was used, after centring
to a fixed reference set (section 2.1). They do not name the
implementation or the reference samples. Other SCAN-B papers may say.

**The experiment's design.** The labels are not recomputed. Only the
selected genes are compared with the 50 PAM50 genes. Each method ranks
all genes once per split. Walking down the ranking, we record how many
PAM50 genes are found at each `k`. The result is a recall curve per
method. For the random control, the expected recall is the straight line
`k` / number of genes. A strong method reaches the full list at a small
`k`. The results are in [section 7.6](07_results_quality.md#76-does-selection-find-the-pam50-genes).

Three difficulties were known in advance:

- **Scoring.** The `k` that reaches all 50 genes is set by the single
  weakest gene. That gene may rank thousands deep for every method. So we
  report the whole curve, and the ranks at which 50% and 90% of the genes
  are found.
- **Methods without a full ranking.** FDR, FWE, Lasso without a fixed
  count, and Higher Criticism return a set, not a ranking. Each gives one
  point, not a curve.
- **Correlated substitutes.** A method may pick a gene highly correlated
  with a PAM50 gene. It finds the same signal, but the pick does not count
  toward recall. Section 2.1 ("Signal is everywhere") explains why this is
  common here.

The experiment needs a gene set that holds all 50 genes. GSE96058 does.
`scanb_small` did not (section 2.1).

## C.3 Higher Criticism and the share of signal

HC searches only the smallest 10% of p-values (alpha0 = 0.1). On SCAN-B the
HC score is still rising at that edge. HC2008 kept 2,855 to 2,856 genes on
every task and split. HC2004 kept 1 to 6 genes, with a median of 1.

HC first scored genes by the F-test. The comparison never ran HC with the
F-test. The switch to the U test was checked in a one-split test on
SCAN-B. In that check, HC2008 still kept 2,855 genes on every task. But
9% to 14% of them changed, and its macro-F1 moved from 0.871 to 0.873.
HC2004 kept 1 gene on every task, as before, except 4 on LumA. Its
macro-F1 moved from 0.744 to 0.707. One split cannot tell these changes
from noise.

BreastMNIST gave a second check. Its signal is not rare either
(chapter 2). HC2008 again kept the edge, 204 features, on every split. So
on both datasets the search limit, not the data, sets HC2008's count.

## C.4 A better model for floor-heavy genes (not built)

The U test merges two ways a gene can differ between classes. One is how
often it is detected at all. The other is how high it is when detected.

A better model describes each gene by four numbers. Two are the class
sizes, n₁ and n₂, given by the dataset. The other two, p₁ and p₂, are the
share of each class whose value sits at the floor. This fits what we see.
51% of the values in the first 200 genes are at the floor. So the floor
is not an occasional tie. It is the main feature of the data.

Under the null hypothesis, the labels are assigned at random. Given the
class sizes and the total number of floor values, the floor values split
between the classes by a hypergeometric draw. That draw sets p₁ and p₂.
The detected values fall in a random order. Together these give an exact
null distribution for each gene. From it come exact tail p-values, where
the normal approximation is weakest. The model needs a separate
computation for each floor count. It is beyond the project's scope and
was not built.

## C.5 Gene counts

- GSE96058 has 30,865 genes, and all 50 PAM50 genes are among them.
- Across all 3,409 GSE96058 samples, 2,236 genes have a single value.
  Across the 2,976 patients we keep, 2,313 do. They are dropped, which
  leaves 28,552.
- 11 more genes are flat: (max − min) / max |x| is below 0.01. The rest are
  at 0.057 or above, so the cutoff falls in a gap. Nine of the 11 have at
  least 2,864 patients at their minimum. Dropping them leaves 28,541.

## C.6 Open questions

- Which target is primary. The specification asks for "the clinical
  class" and does not name PAM50. We chose PAM50. A measured target, such
  as the pathologist's ER status, is listed as open work (chapter 10).
