# Results

Draft. The protocol of chapter 4: 16 splits on SCAN-B and 20 on
BreastMNIST, k from 5 to 1000 on SCAN-B and to 250 on BreastMNIST, RFE at
step 0.5, a forest of 50 trees. Random selection and Mann-Whitney also run
at the count Higher Criticism 2008 chose, to compare it at equal size.

The tables are generated from the runner's results by
`python scripts/report_tables.py docs/report/06_results.md`; no number in
them is typed by hand. Each cell is macro-F1, averaged over tasks within a
split, then the median over the splits with the first and third quartiles
in brackets. The
full tables, per task, with stability and timing, are in
`docs/comparison_scanb.md` and `docs/comparison_breastmnist.md`.

## SCAN-B

Macro-F1, mean over the five subtype-vs-rest tasks. "own" is the count a
method chose itself.

<!-- generated: scanb_macro_f1 -->
| selector | 5 | 10 | 20 | 50 | 100 | 250 | 500 | 1000 | 2855 | own |
|---|---|---|---|---|---|---|---|---|---|---|
| RandomSelector | 0.518 [0.500, 0.534] | 0.575 [0.554, 0.599] | 0.647 [0.637, 0.669] | 0.730 [0.717, 0.742] | 0.775 [0.765, 0.784] | 0.819 [0.813, 0.828] | 0.845 [0.841, 0.850] | 0.855 [0.851, 0.861] | 0.868 [0.865, 0.870] |  |
| MannWhitneySelector | 0.780 [0.771, 0.790] | 0.806 [0.802, 0.814] | 0.826 [0.817, 0.830] | 0.847 [0.842, 0.853] | 0.856 [0.851, 0.860] | 0.868 [0.864, 0.870] | 0.873 [0.870, 0.877] | 0.879 [0.875, 0.883] | 0.883 [0.880, 0.886] |  |
| LassoSelector | 0.778 [0.769, 0.788] | 0.820 [0.809, 0.833] | 0.844 [0.834, 0.849] | 0.845 [0.840, 0.852] | 0.846 [0.842, 0.853] | 0.846 [0.840, 0.851] | 0.847 [0.843, 0.852] | 0.848 [0.842, 0.854] |  |  |
| TreeImportanceSelector | 0.805 [0.794, 0.812] | 0.826 [0.824, 0.830] | 0.844 [0.839, 0.846] | 0.858 [0.854, 0.863] | 0.868 [0.865, 0.872] | 0.876 [0.872, 0.880] | 0.878 [0.876, 0.880] | 0.879 [0.874, 0.882] |  |  |
| RFESelector | 0.604 [0.580, 0.636] | 0.662 [0.645, 0.682] | 0.704 [0.695, 0.708] | 0.744 [0.738, 0.749] | 0.775 [0.770, 0.779] | 0.818 [0.812, 0.822] | 0.835 [0.832, 0.837] | 0.846 [0.843, 0.850] |  |  |
| HC2004 |  |  |  |  |  |  |  |  |  | 0.638 [0.633, 0.645] |
| HC2008 |  |  |  |  |  |  |  |  |  | 0.883 [0.880, 0.886] |
<!-- end generated -->

Gain over random selection at the same k and split:

<!-- generated: scanb_gain -->
| selector | 5 | 10 | 20 | 50 | 100 | 250 | 500 | 1000 | 2855 |
|---|---|---|---|---|---|---|---|---|---|
| MannWhitneySelector | 0.260 [0.240, 0.276] | 0.233 [0.204, 0.255] | 0.175 [0.149, 0.192] | 0.114 [0.101, 0.126] | 0.078 [0.074, 0.094] | 0.044 [0.035, 0.052] | 0.028 [0.023, 0.034] | 0.024 [0.021, 0.027] | 0.017 [0.014, 0.018] |
| LassoSelector | 0.263 [0.242, 0.281] | 0.239 [0.221, 0.266] | 0.189 [0.177, 0.203] | 0.114 [0.102, 0.125] | 0.074 [0.065, 0.088] | 0.024 [0.018, 0.030] | 0.001 [-0.003, 0.011] | -0.007 [-0.012, -0.001] |  |
| TreeImportanceSelector | 0.290 [0.265, 0.300] | 0.252 [0.227, 0.275] | 0.196 [0.175, 0.208] | 0.128 [0.113, 0.139] | 0.093 [0.087, 0.106] | 0.052 [0.049, 0.057] | 0.032 [0.027, 0.038] | 0.025 [0.017, 0.028] |  |
| RFESelector | 0.085 [0.065, 0.105] | 0.082 [0.061, 0.116] | 0.056 [0.036, 0.077] | 0.014 [-0.003, 0.033] | 0.010 [-0.013, 0.015] | -0.003 [-0.015, 0.005] | -0.012 [-0.016, -0.002] | -0.008 [-0.013, -0.005] |  |
<!-- end generated -->

Higher Criticism against fixed-k methods at its own count, same task and
split:

<!-- generated: scanb_own_count -->
| selector | MannWhitneySelector, k = 2855 | RandomSelector, k = 2855 |
|---|---|---|
| HC2008 | 0.000 [0.000, 0.000] | 0.017 [0.014, 0.018] |
<!-- end generated -->

**Selection beats chance most when the budget is small.** At k = 5 every
selector but RFE is far above random. The gain shrinks with every step of
`k`. By k = 250 it is a few hundredths: 250 random genes of
<!-- fact: scanb_loaded_genes -->28,552<!-- /fact --> already carry most of the signal.

**Tree importance leads up to k = 250.** It has the highest median at every
`k` up to 250. Its quartiles lie wholly above those of Mann-Whitney and
Lasso at k = 5, 50, 100 and 250. From k = 500 Mann-Whitney catches up.

**Lasso stops improving at k = 20.** Its penalty leaves only a few hundred
genes with a non-zero weight. At k = 500 and 1000 it keeps fewer genes than
asked, and at k = 1000 it falls below random selection.

**RFE is the weakest selector.** Its gain is clearly above zero only up to
k = 20. From k = 500 its quartiles lie below zero: it does worse than
random.

**Higher Criticism 2008 is a top-k filter at the edge of its search.** It
keeps a tenth of the genes, the most its search allows, on every task and
split. At that count it scores exactly as Mann-Whitney does, because both
rank genes by the same U test. Its score looks highest only because it uses
far more genes than any fixed `k`. Random selection of as many genes comes
close behind it. HC2004 keeps one gene on most tasks and scores far lower.

## BreastMNIST

Macro-F1, malignant against benign or normal:

<!-- generated: breastmnist_macro_f1 -->
| selector | 5 | 10 | 20 | 50 | 100 | 204 | 250 | own |
|---|---|---|---|---|---|---|---|---|
| RandomSelector | 0.429 [0.422, 0.456] | 0.468 [0.432, 0.495] | 0.521 [0.463, 0.567] | 0.576 [0.534, 0.653] | 0.675 [0.619, 0.699] | 0.739 [0.717, 0.760] | 0.747 [0.732, 0.765] |  |
| MannWhitneySelector | 0.640 [0.613, 0.674] | 0.681 [0.666, 0.698] | 0.714 [0.692, 0.731] | 0.748 [0.734, 0.761] | 0.773 [0.756, 0.789] | 0.790 [0.771, 0.810] | 0.795 [0.782, 0.817] |  |
| LassoSelector | 0.668 [0.642, 0.689] | 0.700 [0.675, 0.716] | 0.736 [0.727, 0.748] | 0.778 [0.757, 0.789] | 0.798 [0.780, 0.816] |  | 0.815 [0.796, 0.827] |  |
| TreeImportanceSelector | 0.641 [0.621, 0.686] | 0.706 [0.687, 0.730] | 0.745 [0.732, 0.757] | 0.761 [0.740, 0.779] | 0.773 [0.766, 0.788] |  | 0.793 [0.785, 0.807] |  |
| RFESelector | 0.607 [0.575, 0.656] | 0.653 [0.620, 0.694] | 0.702 [0.680, 0.715] | 0.752 [0.739, 0.773] | 0.790 [0.774, 0.802] |  | 0.803 [0.797, 0.812] |  |
| HC2004 |  |  |  |  |  |  |  | 0.521 [0.498, 0.583] |
| HC2008 |  |  |  |  |  |  |  | 0.790 [0.771, 0.810] |
<!-- end generated -->

Gain over random selection at the same k and split:

<!-- generated: breastmnist_gain -->
| selector | 5 | 10 | 20 | 50 | 100 | 204 | 250 |
|---|---|---|---|---|---|---|---|
| MannWhitneySelector | 0.192 [0.173, 0.239] | 0.201 [0.173, 0.250] | 0.199 [0.134, 0.242] | 0.167 [0.100, 0.208] | 0.110 [0.063, 0.146] | 0.060 [0.037, 0.084] | 0.059 [0.026, 0.063] |
| LassoSelector | 0.223 [0.188, 0.244] | 0.236 [0.188, 0.254] | 0.210 [0.181, 0.279] | 0.192 [0.127, 0.241] | 0.129 [0.115, 0.164] |  | 0.066 [0.045, 0.079] |
| TreeImportanceSelector | 0.209 [0.183, 0.238] | 0.231 [0.202, 0.259] | 0.232 [0.156, 0.264] | 0.181 [0.110, 0.237] | 0.106 [0.083, 0.150] |  | 0.056 [0.038, 0.060] |
| RFESelector | 0.156 [0.125, 0.207] | 0.177 [0.131, 0.234] | 0.164 [0.114, 0.242] | 0.161 [0.130, 0.202] | 0.118 [0.099, 0.155] |  | 0.048 [0.042, 0.070] |
<!-- end generated -->

Higher Criticism against fixed-k methods at its own count:

<!-- generated: breastmnist_own_count -->
| selector | MannWhitneySelector, k = 204 | RandomSelector, k = 204 |
|---|---|---|
| HC2008 | 0.000 [0.000, 0.000] | 0.060 [0.037, 0.084] |
<!-- end generated -->

The main pattern holds on image embeddings. In the median, every selector
beats random at every `k`, most at k = 5 to 20. Lasso or tree importance
has the highest median at every `k`; their quartiles overlap throughout.
RFE is again weakest at small `k`, but here it catches up from k = 100.
HC2008 again stops at the edge of its search and again equals Mann-Whitney
at that count. Its gain over random selection of as many features is small
but above zero across its quartiles.

## Limits of these results

- The quartiles measure re-splitting the same patients, not new patients.
- RFE uses step 0.5, chosen for speed. Step 0.1 gave about the same scores
  on SCAN-B in an earlier run.
- The forest has 50 trees, chosen for speed. On BreastMNIST 200 trees gave
  about the same scores, with more stable choices.
- Lasso keeps only features with a non-zero weight. On SCAN-B it keeps a
  few hundred genes at most, so its cells at k = 500 and 1000 hold fewer
  genes than `k`. On BreastMNIST it keeps fewer than 250 at k = 250.
- HC2008's count is set by its search limit on both datasets. Chapter 3
  explains why: neither dataset has rare signal.
