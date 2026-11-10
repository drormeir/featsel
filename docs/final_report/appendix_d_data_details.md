# Appendix D. Data: plans, open questions and extra facts

[Back to contents](README.md)

Chapter 2 describes the data. This appendix keeps the plans that were made
about it, what is still open, and facts too detailed for the chapter.

## D.1 Reproducing the ER label (planned, not done)

Decided 2026-10-17. ER is a target in this project, as in the earlier
course report. Unlike that report, this one states where the ER label comes
from, and plans to reproduce it.

1. Rebuild Brueffer et al.'s ER multigene classifier. Their Supplementary
   Table 3 gives its shrunken centroids per gene: 459 are non-zero for ER.
   The classifier also needs gene means and spreads. These come from the
   GSE81538 training cohort: 405 patients, with the pathologists'
   consensus labels.
2. Apply it to the GSE96058 samples. Compare its calls with
   `er prediction mgc` and with their Supplementary Table 8.
3. Report the agreement. Also report any settings that had to be inferred,
   such as the offset and class priors of the nearest shrunken centroid
   method.

The paper and its Data Supplement are in `references/local/`. GSE81538 is
not downloaded yet.

## D.2 The source of PAM50, and recovering its genes

Decided 2026-10-17. The report discusses where the PAM50 label comes from,
and tests whether feature selection recovers the genes that define it.

**Still open.** How SCAN-B assigned `pam50 subtype`: which implementation,
and on which data. Other SCAN-B papers may say.

**The experiment's design.** The labels are not recomputed; only the gene
subset is compared. Each method ranks all genes once per split. Walking
down the ranking, we record how many PAM50 genes are found at each `k`.
The result is a recall curve per method. The random control is the
straight line `k` / number of genes. A strong method reaches the full list
at a small `k`. The results are in [section 7.6](07_results_quality.md#76-does-selection-find-the-pam50-genes).

Three difficulties were known in advance:

- **Scoring.** The `k` that reaches all 50 genes is set by the single
  weakest gene. That gene may rank thousands deep for every method. So we
  report the whole curve, and the rank at which 50% and 90% are found.
- **Methods without a full ranking.** FDR, FWE, Lasso without a fixed
  count, and Higher Criticism return a set, not a ranking. They give one
  point, not a curve.
- **Correlated substitutes.** A method that picks a gene highly correlated
  with a PAM50 gene finds the same signal, but scores no recall. Section
  2.1 ("Signal is everywhere") explains why this is common here.

The experiment needs a gene set that holds all 50 genes. GSE96058 does;
the retired `scanb_small` did not.

## D.3 Higher Criticism and the share of signal

HC searches only the smallest 10% of p-values (alpha0 = 0.1). On SCAN-B the
HC score is still rising at that edge. HC2008 kept 2,855 to 2,856 genes on
every task and split. HC2004 kept 1 to 6 genes, with a median of 1.

The first runs scored genes by the F-test, and HC behaved the same way.
On 2026-10-31 the comparison was rerun with the U test. The change of test
changed nothing.

BreastMNIST was the place to check whether HC behaves differently. It was
checked on 2026-10-31: its signal is not rare either (chapter 2). HC2008
again kept the edge, 204 features, on every split. So on both datasets the
search limit, not the data, sets HC2008's count.

## D.4 A better model for floor-heavy genes (not built)

The U test merges two ways a gene can differ between classes. One is how
often it is detected at all. The other is how high it is when detected.

A better model describes each gene by four numbers. Two are the class
sizes, n₁ and n₂, given by the dataset. The other two, p₁ and p₂, are the
share of each class whose value sits at the floor. This fits what we see.
51% of the values in the first 200 genes are at the floor. So the floor
is not an occasional tie. It is the main feature of the data.

Under the null hypothesis, given the four numbers, the labels split the
floor values by a hypergeometric draw. They split the detected values by a
random order. That gives an exact null distribution for each gene, and so
exact tail p-values, where the normal approximation is weakest. It needs a
separate computation per floor count. It is beyond the project's scope and
not built.

## D.5 Gene counts

- GSE96058 has 30,865 genes, and all 50 PAM50 genes are among them.
- Across all 3,409 GSE96058 samples, 2,236 genes have a single value.
  Across the 2,976 patients we keep, 2,313 do. They are dropped, which
  leaves 28,552.
- `scanb_small`, the course's smaller copy, has 9,264 genes and lacks 11
  PAM50 genes (section 2.1).

## D.6 Open questions

- How the course chose the 9,264 genes of `scanb_small`.
- How much the repeat samples inflate results computed on the course's
  copies.
- Which target is primary. The specification asks for "the clinical
  class" and does not name PAM50. We chose PAM50; a measured target such
  as the pathologist's ER status is listed as open work (chapter 10).
