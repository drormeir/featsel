# Appendix C. Data: plans, open questions and extra facts

[Back to contents](README.md)

Chapter 2 describes the data. This appendix keeps the plans, the open
questions and the finer facts.

## C.1 Reproducing the ER label (planned, not done)

ER was planned as a target, but no experiment uses it. Chapter 2 gives
the label's source. The plan to reproduce the published classifier
behind it:

1. Rebuild Brueffer et al.'s ER multigene classifier from its shrunken
   centroids per gene (their Supplementary Table 3). For ER, 459 genes
   have a non-zero centroid. The gene means and spreads come from the
   GSE81538 training cohort: 405 patients, with the pathologists'
   consensus labels.
2. Apply it to the GSE96058 samples. Compare its calls with
   `er prediction mgc` and with their Supplementary Table 8.
3. Report the agreement and any settings that had to be inferred, such as
   the offset and class priors of the nearest shrunken centroid method.

Known risks:

- GSE81538 was mapped to an older human genome (hg19) than GSE96058
  (hg38). So a few gene names may not match.
- The 459 genes were chosen automatically from the 5,000 most variable
  genes of the 405 training patients. A rebuild from scratch with another
  shrinkage threshold would give a different list.

## C.2 The source of PAM50, and recovering its genes

Chapter 2 gives the source of the PAM50 label. Section 7.6 tests whether
feature selection recovers its genes.

**Still open.** How SCAN-B assigned `pam50 subtype`. The GEO notes say
only that an implementation of Parker's method was used, after centring
to a fixed reference set (section 2.1). They name neither the
implementation nor the reference samples. Other SCAN-B papers may say.

**The experiment's design.** The labels are not recomputed. Only the
selected genes are compared with the 50 PAM50 genes. Each method ranks
all genes once per split. Counting the PAM50 genes down the ranking gives
a recall curve over `k`. For the random control, the expected recall is
the line `k` / number of genes. A strong method reaches the full list at
a small `k`. The results are in
[section 7.6](07_results_quality.md#76-does-selection-find-the-pam50-genes).

Known difficulties:

- **Scoring.** The `k` that reaches all 50 genes is set by the weakest
  gene. It may rank thousands deep for every method. So we report the
  whole curve, and the ranks at which 50% and 90% of the genes are found.
- **Methods without a full ranking.** FDR, FWE, Lasso without a fixed
  count, and Higher Criticism return a set. Each gives one point, not a
  curve.
- **Correlated substitutes.** A gene highly correlated with a PAM50 gene
  carries the same signal, but does not count toward recall. Section 2.1
  ("Signal is everywhere") explains why this is common here.

The experiment needs a gene set with all 50 genes. GSE96058 has them.
`scanb_small` did not (section 2.1).

## C.3 Higher Criticism and the share of signal

HC searches only the smallest 10% of p-values (alpha0 = 0.1). On SCAN-B,
the HC score is still rising at that edge. HC2008 kept 2,855 to 2,856
genes on every task and split. HC2004 kept 1 to 6 genes, with a median
of 1.

HC first scored genes by the F-test. The comparison never ran HC with the
F-test. A one-split test on SCAN-B checked the switch to the U test:

| | Genes kept with the U test | Macro-F1, F-test → U test |
|---|---|---|
| HC2008 | 2,855 on every task, as before; 9% to 14% of them changed | 0.871 → 0.873 |
| HC2004 | 1 on every task, as before, except 4 on LumA | 0.744 → 0.707 |

One split cannot tell these changes from noise.

BreastMNIST gave a second check: its signal is not rare either
(chapter 2). HC2008 again kept the edge, 204 features, on every split. So
on both datasets the search limit, not the data, sets HC2008's count.

## C.4 A better model for floor-heavy genes (not built)

The U test merges two ways a gene can differ between classes: how often
it is detected, and how high it is when detected.

A better model describes each gene by four numbers: the class sizes n₁
and n₂, given by the dataset, and p₁ and p₂, the share of each class at
the floor. This fits the data. 51% of the values in the first 200 genes
are at the floor (section 2.1). So the floor is the main feature of the
data, not an occasional tie.

Under the null hypothesis, the labels are random. Given the class sizes
and the total floor count, the floor values split between the classes by
a hypergeometric draw. That draw sets p₁ and p₂. The detected values fall
in a random order. Together these give an exact null distribution for
each gene, and exact tail p-values where the normal approximation is
weakest. The model needs a separate computation for each floor count. It
is beyond the project's scope and was not built.

## C.5 Gene counts

- GSE96058 has 30,865 genes, including all 50 PAM50 genes.
- 2,236 genes have a single value across all 3,409 GSE96058 samples, and
  2,313 across the 2,976 kept patients. Dropping the 2,313 leaves 28,552.
- 11 more genes are flat: (max − min) / max |x| is below 0.01. All other
  genes are at 0.057 or above, so the cutoff falls in a gap. Nine of the
  11 have at least 2,864 patients at their minimum. Dropping them leaves
  28,541.

## C.6 Open questions

- Which target is primary. The specification asks for "the clinical
  class" and does not name PAM50. We chose PAM50. A measured target, such
  as the pathologist's ER status, is open work (chapter 10).
