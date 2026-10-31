# Results

Draft. First complete run of the protocol in chapter 4, at its smallest
size: 5 splits, k = 10, 50 and 250, RFE at step 0.5. Every number here
comes from `docs/comparison_scanb.md` and `docs/comparison_breastmnist.md`,
which the summary script generates from the runner's output. Each number is
the median over the 5 splits.

## SCAN-B

Macro-F1, mean over the five subtype-vs-rest tasks:

| Method | k = 10 | k = 50 | k = 250 | Own count |
|---|---|---|---|---|
| Random (control) | 0.548 | 0.727 | 0.828 | |
| Mann-Whitney | 0.813 | 0.845 | 0.869 | |
| Lasso | 0.829 | 0.846 | 0.851 | |
| Tree importance | 0.830 | 0.860 | 0.877 | |
| RFE | 0.664 | 0.733 | 0.814 | |
| HC 2004 | | | | 0.637 (1 gene) |
| HC 2008 | | | | 0.884 (2,855 genes) |

**Selection beats chance most when the budget is small.** At k = 10,
Mann-Whitney, Lasso and tree importance beat the random control by 0.25 to
0.28 macro-F1. At k = 250 the gain falls to 0.02 to 0.05: 250 random genes
of 28,552 already carry most of the signal.

**RFE is the weakest selector here.** It beats random by 0.12 at k = 10 and
not at all at k = 250. It fails on LumA (0.46 at k = 10) and Normal (0.51).

**The two Higher Criticism variants behave very differently.** HC2004 keeps
one gene. HC2008 keeps 2,855 or 2,856 genes on every task and split, and scores
best overall.
2,855 is the edge of its search, a tenth of the genes, so the search limit
sets its count, not the data.

**The gain differs by subtype.** At k = 10 Basal is easy for every
selector but random (0.95 to 0.96). Normal is hardest: 0.67 to 0.71 for
Mann-Whitney, Lasso and tree importance.

**Stability.** Mann-Whitney (Kuncheva 0.68 to 0.84) and HC2008 (0.87) pick
nearly the same genes on every split. Lasso, tree importance and RFE do not
(0.19 to 0.43), although they score as well as Mann-Whitney.

## BreastMNIST

Macro-F1, malignant against benign or normal:

| Method | k = 10 | k = 50 | k = 250 | Own count |
|---|---|---|---|---|
| Random (control) | 0.468 | 0.613 | 0.740 | |
| Mann-Whitney | 0.683 | 0.762 | 0.784 | |
| Lasso | 0.709 | 0.786 | 0.816 | |
| Tree importance | 0.717 | 0.772 | 0.802 | |
| RFE | 0.627 | 0.748 | 0.800 | |
| HC 2004 | | | | 0.522 (1 dimension) |
| HC 2008 | | | | 0.778 (204 dimensions) |

The same pattern holds on image embeddings. Every selector beats random at
every k, by 0.16 to 0.28 at k = 10 and 0.06 to 0.07 at k = 250. RFE is
again weakest at k = 10, but here it catches up by k = 250. HC2008 again
stops at the edge of its search, 204 of 2,048 dimensions.

## Limits of these results

- Five splits give a rough spread; the ranges are in the source tables.
  Many differences between the non-random selectors lie within them.
- RFE uses step 0.5, chosen for speed. Step 0.1 gave about the same scores
  on SCAN-B in an earlier run.
- Lasso keeps only genes with a non-zero weight, as few as 155 for Basal at
  k = 250 on SCAN-B and 162 to 182 on BreastMNIST.
- HC2008's count is set by its search limit on both datasets. That limit
  is examined in chapter 3 and still open.
