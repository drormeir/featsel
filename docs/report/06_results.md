# Results

Draft. First complete run of the protocol in chapter 4, at its smallest
size: 5 splits, k = 10, 50 and 250, RFE at step 0.5. Random selection and
Mann-Whitney also run at the count Higher Criticism 2008 chose, to compare
it at equal size.

The tables are generated from the runner's results by
`python scripts/report_tables.py docs/report/06_results.md`; no number in
them is typed by hand. Each cell is macro-F1, averaged over tasks within a
split, then the median over the 5 splits with the range in brackets. The
full tables, per task, with stability and timing, are in
`docs/comparison_scanb.md` and `docs/comparison_breastmnist.md`.

## SCAN-B

Macro-F1, mean over the five subtype-vs-rest tasks. "own" is the count a
method chose itself.

<!-- generated: scanb_macro_f1 -->
| selector | 10 | 50 | 250 | 2855 | own |
|---|---|---|---|---|---|
| RandomSelector | 0.548 [0.516, 0.600] | 0.727 [0.685, 0.753] | 0.828 [0.813, 0.846] | 0.870 [0.866, 0.874] |  |
| MannWhitneySelector | 0.813 [0.801, 0.820] | 0.845 [0.844, 0.856] | 0.869 [0.861, 0.879] | 0.884 [0.879, 0.895] |  |
| TreeImportanceSelector | 0.830 [0.807, 0.843] | 0.860 [0.844, 0.865] | 0.877 [0.874, 0.887] |  |  |
| HC2004 |  |  |  |  | 0.637 [0.632, 0.643] |
| HC2008 |  |  |  |  | 0.884 [0.879, 0.895] |
| LassoSelector | 0.829 [0.810, 0.840] | 0.846 [0.830, 0.851] | 0.851 [0.841, 0.867] |  |  |
| RFESelector | 0.664 [0.632, 0.684] | 0.733 [0.718, 0.792] | 0.814 [0.812, 0.835] |  |  |
<!-- end generated -->

Gain over random selection at the same k and split:

<!-- generated: scanb_gain -->
| selector | 10 | 50 | 250 | 2855 |
|---|---|---|---|---|
| MannWhitneySelector | 0.254 [0.215, 0.297] | 0.123 [0.092, 0.170] | 0.040 [0.023, 0.053] | 0.014 [0.011, 0.021] |
| TreeImportanceSelector | 0.273 [0.243, 0.316] | 0.133 [0.099, 0.180] | 0.050 [0.040, 0.061] |  |
| LassoSelector | 0.281 [0.218, 0.317] | 0.123 [0.080, 0.160] | 0.021 [0.006, 0.041] |  |
| RFESelector | 0.116 [0.057, 0.167] | 0.019 [-0.023, 0.065] | -0.014 [-0.033, 0.022] |  |
<!-- end generated -->

Higher Criticism against fixed-k methods at its own count, same task and
split:

<!-- generated: scanb_own_count -->
| selector | MannWhitneySelector, k = 2855 | RandomSelector, k = 2855 |
|---|---|---|
| HC2008 | 0.000 [0.000, 0.000] | 0.014 [0.011, 0.021] |
<!-- end generated -->

**Selection beats chance most when the budget is small.** At k = 10 every
selector but RFE is far above random. By k = 250 the gain has shrunk to a
few hundredths: 250 random genes of 28,552 already carry most of the
signal.

**RFE is the weakest selector here.** At k = 10 its whole range lies below
the other selectors'; at k = 250 it is no better than random.

**Higher Criticism 2008 is a top-k filter at the edge of its search.** It
keeps a tenth of the genes, the most its search allows, on every task and
split. At that count it scores exactly as Mann-Whitney does, because both
rank genes by the same U test. Its score looks highest only because it uses
far more genes than k = 250. Random selection of as many genes comes close
behind it. HC2004 keeps one gene and scores far lower.

**The ranking among Mann-Whitney, Lasso and tree importance is mostly not
settled.** Their ranges overlap at k = 10 and 50. At k = 250 tree
importance's range lies wholly above Lasso's.

## BreastMNIST

Macro-F1, malignant against benign or normal:

<!-- generated: breastmnist_macro_f1 -->
| selector | 10 | 50 | 204 | 250 | own |
|---|---|---|---|---|---|
| RandomSelector | 0.468 [0.432, 0.520] | 0.613 [0.536, 0.677] | 0.754 [0.730, 0.788] | 0.740 [0.719, 0.751] |  |
| MannWhitneySelector | 0.683 [0.666, 0.696] | 0.762 [0.704, 0.788] | 0.778 [0.766, 0.827] | 0.784 [0.778, 0.818] |  |
| LassoSelector | 0.709 [0.696, 0.756] | 0.786 [0.763, 0.792] |  | 0.816 [0.784, 0.851] |  |
| TreeImportanceSelector | 0.717 [0.681, 0.751] | 0.772 [0.750, 0.805] |  | 0.802 [0.783, 0.810] |  |
| RFESelector | 0.627 [0.606, 0.697] | 0.748 [0.722, 0.772] |  | 0.800 [0.785, 0.811] |  |
| HC2004 |  |  |  |  | 0.522 [0.485, 0.645] |
| HC2008 |  |  |  |  | 0.778 [0.766, 0.827] |
<!-- end generated -->

Gain over random selection at the same k and split:

<!-- generated: breastmnist_gain -->
| selector | 10 | 50 | 204 | 250 |
|---|---|---|---|---|
| MannWhitneySelector | 0.204 [0.176, 0.251] | 0.144 [0.076, 0.252] | 0.024 [-0.022, 0.089] | 0.059 [0.027, 0.078] |
| LassoSelector | 0.277 [0.176, 0.305] | 0.174 [0.112, 0.255] |  | 0.069 [0.047, 0.131] |
| TreeImportanceSelector | 0.257 [0.179, 0.318] | 0.159 [0.113, 0.269] |  | 0.063 [0.051, 0.073] |
| RFESelector | 0.158 [0.101, 0.264] | 0.146 [0.066, 0.212] |  | 0.063 [0.048, 0.079] |
<!-- end generated -->

Higher Criticism against fixed-k methods at its own count:

<!-- generated: breastmnist_own_count -->
| selector | MannWhitneySelector, k = 204 | RandomSelector, k = 204 |
|---|---|---|
| HC2008 | 0.000 [0.000, 0.000] | 0.024 [-0.022, 0.089] |
<!-- end generated -->

The same pattern holds on image embeddings. In the median, every selector
beats random at every k, most at k = 10. RFE is again weakest at k = 10, but here it
catches up by k = 250. HC2008 again stops at the edge of its search and
again equals Mann-Whitney at that count; its gain over random selection of
as many features is small and its range includes zero.

## Limits of these results

- Five splits give a rough spread. It measures re-splitting the same
  patients, not new patients.
- RFE uses step 0.5, chosen for speed. Step 0.1 gave about the same scores
  on SCAN-B in an earlier run.
- Lasso keeps only features with a non-zero weight. At k = 250 it keeps
  fewer than 250 on the Basal task of SCAN-B, and on BreastMNIST.
- HC2008's count is set by its search limit on both datasets. Chapter 3
  explains why: neither dataset has rare signal.
