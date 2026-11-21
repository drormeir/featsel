# 8. Results on image data: BreastMNIST and FGVC-Aircraft

[Back to contents](README.md) · Previous: [7. Results on SCAN-B](07_results_scanb.md) · Next: [9. Conclusions](09_conclusions.md)

This chapter reports the two image datasets. Each image is a ResNet-50
embedding of 2,048 features (chapter 2). BreastMNIST is task 4. FGVC-Aircraft
is a dataset the pipeline never saw during its design. Section 8.11 adds
simulated data with rare signal, the setting Higher Criticism was built
for.

Every run follows the protocol of chapter 3. All numbers are means over
the 16 splits.

Sources: [`scripts/summarize_frontier.py`](../scripts/summarize_frontier.py)
wrote the `summary_*` tables in [`figures/`](figures/), and
[`scripts/paired_tests.py`](../scripts/paired_tests.py) the `paired_tests_*`
tables. Configs: `configs/frontier_breastmnist70_nocopies.yaml` and
[`configs/aircraft70_gpu.yaml`](../configs/aircraft70_gpu.yaml).

## 8.1 BreastMNIST: what each method reaches

585 images, without the copied ones (section 8.8): 409 in the training set
and 176 in the test set. Malignant 180, benign or normal 405.

![Frontier on BreastMNIST](figures/frontier_breastmnist70_nocopies_roc_auc.png)

*Each point is one method at one `k`: median selection time per fit, and
mean test AUC with one standard deviation over the splits. Table:
[`figures/frontier_breastmnist70_nocopies_roc_auc.csv`](figures/frontier_breastmnist70_nocopies_roc_auc.csv).*

Each method at the `k` it chose on validation sets:

| Method | Test AUC [Q1, Q3] | Features kept | Selection, s per fit | Search, s |
|---|---|---|---|---|
| All features | **0.895** [0.877, 0.906] | 2,048 | none | none |
| RFE | **0.891** [0.875, 0.906] | 500 to 1,000 | 0.12 | 0.8 |
| Mutual information | 0.887 [0.873, 0.901] | 500 to 1,000 | 0.072 | 0.8 |
| Tree importance | 0.885 [0.872, 0.894] | 500 to 1,000 | 0.37 | 2.4 |
| RFECV, scored by AUC | 0.884 [0.863, 0.902] | 212 to 1,232 | 0.10 | 1.0 |
| Mann-Whitney | 0.883 [0.869, 0.896] | 500 to 1,000 | 0.008 | 0.4 |
| Lasso | 0.882 [0.868, 0.898] | 73 to 182 | 0.084 | 2.0 |
| Correlation (Spearman) | 0.866 [0.848, 0.881] | 10 to 813 | 0.020 | 0.5 |
| HC2008 | 0.848 [0.839, 0.867] | 204 | 0.008 | none |
| RFECV, scored by accuracy | 0.847 [0.816, 0.885] | 8 to 416 | 0.11 | 0.8 |
| Berk-Jones | 0.847 [0.839, 0.866] | 175 to 204 | 0.007 | none |
| HC2004 | 0.771 [0.752, 0.790] | 1 to 2 | 0.008 | none |

Search is the time to choose `k` on validation sets, or Lasso's C and
RFECV's count. Sequential selection has not yet run on the images without
copies (section 10.1). Correlation, mutual information, the chains and
both RFECV were added after the first run, on the same splits. Table:
[`figures/summary_breastmnist70_nocopies_chosen.csv`](figures/summary_breastmnist70_nocopies_chosen.csv).

What it shows:

- **Every method with a `k` list chose the largest `k` it was offered, or
  close to it, in most splits.** Validation AUC kept rising with `k`.
- **No method beat all features.** RFE came closest, 0.003 below.
- **The GPU filters are the cheap choice.** Mann-Whitney reached 0.883 in
  0.008 s per fit. RFE gained 0.009 more for 15 times the time.
- **Correlation equals Mann-Whitney up to `k` = 500.** Spearman correlation
  with a binary label ranks features as the U test does. At `k` = 1000 it
  kept about 800: its redundancy filter dropped the rest.

## 8.2 The bar: random features at each `k`

At small `k`, selection matters. At large `k`, random features catch up.

![Test AUC against k on the image data](figures/curves_images70.png)

*Test AUC at each `k`, mean over the 16 splits, on both image datasets.
The dotted line is all 2,048 features. Plot: [`scripts/plot_curves.py`](../scripts/plot_curves.py).*

Every method's test AUC at every `k` is in Appendix C.2 (BreastMNIST)
and C.3 (FGVC-Aircraft).

The corrected test (section 3.3) against random at the same `k`:

| `k` | Established gains (p < 0.05) |
|---|---|
| 10 | Mutual information +0.163, tree importance +0.160, Mann-Whitney +0.156, correlation +0.156, RFE +0.102; chains +0.117 and +0.156 |
| 50 | Tree importance +0.090, mutual information +0.085; Mann-Whitney 500 then tree importance +0.091 |
| 250 | RFE +0.071; chains +0.050 and +0.051 |
| 500 | none |
| 1000 | none |

At `k` = 10 every method won all 16 splits against random, except RFE and
Mann-Whitney 500 then RFE (15 of 16). From `k` = 500 no gain is
established. Tables:
[`figures/summary_breastmnist70_nocopies_bar.csv`](figures/summary_breastmnist70_nocopies_bar.csv),
[`figures/paired_tests_breastmnist70_nocopies.csv`](figures/paired_tests_breastmnist70_nocopies.csv).

The order between methods may turn with `k`, but here no step of it is
established. At `k` = 10 the filters lead RFE (Mann-Whitney by 0.055,
p = 0.15). At `k` = 250 and 500, RFE leads Mann-Whitney and correlation
(by 0.028, p = 0.19, and by 0.023, p = 0.07). At 1,000 they tie (p = 0.33).
On FGVC-Aircraft the lead of RFE at large `k` is established (section
8.9). A filter scores each feature alone. RFE removes features while it
looks at all of them together, and that pays only once many are kept.

## 8.3 Selection against all features

Each method at its chosen `k`, against all 2,048 features, on the same
splits:

| Method | AUC difference [95% interval] | p | Splits won |
|---|---|---|---|
| RFE | −0.003 [−0.017, 0.010] | 0.59 | 6 of 16 |
| Mutual information | −0.007 [−0.027, 0.013] | 0.45 | 4 |
| Tree importance | −0.009 [−0.028, 0.009] | 0.30 | 4 |
| RFECV, by AUC | −0.011 [−0.034, 0.011] | 0.31 | 2 |
| Mann-Whitney | −0.012 [−0.026, 0.002] | 0.09 | 2 |
| Lasso | −0.013 [−0.036, 0.011] | 0.28 | 3 |
| Correlation | −0.028 [−0.060, 0.003] | 0.07 | 0 |
| Mann-Whitney 500, then RFE | −0.035 [−0.060, −0.010] | 0.009 | 0 |
| Mann-Whitney 500, then Lasso | −0.037 [−0.057, −0.017] | 0.001 | 0 |
| Mann-Whitney 500, then tree importance | −0.040 [−0.073, −0.007] | 0.02 | 1 |
| HC2008 | −0.047 [−0.087, −0.007] | 0.02 | 0 |
| RFECV, by accuracy | −0.048 [−0.122, 0.026] | 0.19 | 1 |
| Berk-Jones | −0.048 [−0.088, −0.008] | 0.02 | 0 |
| HC2004 | −0.123 [−0.169, −0.077] | < 0.001 | 0 |

Table: [`figures/summary_breastmnist70_nocopies_tests.csv`](figures/summary_breastmnist70_nocopies_tests.csv).

The best methods tie all features. The chains, HC2008, Berk-Jones and
HC2004 lose (p < 0.05). On these embeddings, selection buys a smaller
model, not a better one. A model on 1,000 features scores the same as one
on 2,048.

## 8.4 Stability

Nogueira's measure over the 16 splits, at the chosen `k` (section 3.6):

| Method | Stability |
|---|---|
| HC2004 | 0.85 |
| HC2008, Berk-Jones | 0.77 |
| Mann-Whitney | 0.63 |
| Correlation | 0.61 |
| Lasso | 0.34 |
| Tree importance | 0.28 |
| RFE | 0.27 |
| RFECV, by AUC | 0.25 |
| Mutual information | 0.22 |
| RFECV, by accuracy | 0.21 |

The threshold methods, Mann-Whitney and correlation are the most stable
(0.61 to 0.85). Mutual information, RFE, RFECV, Lasso and tree importance
pick a different set on each split (0.21 to 0.34). Except RFECV by
accuracy, they reach 0.882 to 0.891 AUC, against 0.883 for Mann-Whitney,
and none of them is established below all features. So different sets
score about as well.

## 8.5 Higher Criticism and Berk-Jones

HC2008 kept 204 features on every split. That is the edge of its search,
0.1 × 2,048. Berk-Jones kept 175 to 204, so it behaves like HC2008, with
the same AUC and stability. Both lose to Mann-Whitney at 250 features by
about 0.004, and to all features by 0.047 and 0.048.

HC2004 kept 1 or 2 features. It scored 0.771, the lowest of all. Its
statistic peaks at the single strongest p-value (section 4.4).

As on SCAN-B (section 7.5), neither variant found the point where the
signal ends: 27% to 35% of the dimensions pass the Benjamini-Hochberg test
(chapter 2).

## 8.6 RFECV: accuracy against AUC

Scored by accuracy, RFECV kept 8 to 416 features and reached 0.847.
Scored by AUC, it kept 212 to 1,232 and reached 0.884. The difference,
0.037, is not established (p = 0.34;
[`figures/rfecv_scoring_tests.csv`](figures/rfecv_scoring_tests.csv)).
AUC is the safer inner score, as on SCAN-B (section 7.8), but here the
data do not show that it is better.

## 8.7 Two-stage selection

Mann-Whitney first keeps 500 features on the GPU. A slower method then
picks `k` of them, for `k` up to 250.

| Chain | Test AUC | Single method at its own chosen `k` |
|---|---|---|
| Mann-Whitney 500, then RFE | 0.860 | RFE 0.891 |
| Mann-Whitney 500, then Lasso | 0.858 | Lasso 0.882 |
| Mann-Whitney 500, then tree importance | 0.855 | tree importance 0.885 |

Every chain lost to all features (p ≤ 0.021). Every chain also scored below
its second method run alone. The chains save little time here: every method
already runs in under a second on 2,048 features. The cap of 250 also
matters, since every method preferred larger `k`.

**The size of the first step.** The 500 was set by hand, and no reason
was recorded. A chain's `k` stops below its first step, so the 500 also
caps `k` at 250. A side study kept 250 or 1,000 features in the first
step instead, with the same splits and the same second steps, on
BreastMNIST and FGVC-Aircraft (section 8.9). Each size is compared with
500 at the `k` chosen on validation sets, by the corrected test (section
3.3). The time is the mean over splits of each split's median selection
time per fit, 10 fits (3 on FGVC-Aircraft). The main runs take the median
over splits instead.

| Chain | Dataset | Test AUC at 250 / 500 / 1,000 | 1,000 against 500 (p; splits won of 16) | Seconds per fit at 500 / 1,000 |
|---|---|---|---|---|
| then Lasso | BreastMNIST | 0.863 / 0.858 / 0.879 | +0.021 (0.052; 15) | 0.08 / 0.11 |
| then Lasso | FGVC-Aircraft | 0.812 / 0.828 / 0.847 | +0.019 (0.084; 14) | 0.15 / 0.23 |
| then tree importance | BreastMNIST | 0.862 / 0.855 / 0.866 | +0.012 (0.52; 11) | 0.39 / 0.38 |
| then tree importance | FGVC-Aircraft | 0.781 / 0.814 / 0.833 | +0.020 (0.17; 13) | 0.68 / 1.00 |
| then RFE | BreastMNIST | 0.847 / 0.860 / 0.878 | +0.019 (0.21; 12) | 0.12 / 0.13 |
| then RFE | FGVC-Aircraft | 0.821 / 0.829 / 0.854 | +0.025 (0.025; 16) | 0.35 / 0.34 |

*Sources: `chain_size_breastmnist70_nocopies.csv` and
`chain_size_aircraft70.csv` in [`figures/`](figures/) (Appendix A.2).*

At 1,000, every chain gains on both datasets, by 0.012 to 0.025 AUC, for
at most 0.32 s more per fit. One gain is established: RFE on
FGVC-Aircraft. RFE and tree importance chose, as a median over splits,
the largest `k` they were allowed, 250 after 500 and 500 after 1,000. So
most of the gain is likely the larger `k` that a larger first step
allows. Lasso has no such cap and still gained. The main results keep
500: changing it now would let the test set choose the setting. Whether
the chains at 1,000 still lose to all features was not tested.

## 8.8 The copied images

BreastMNIST holds 150 groups of copied images (section 2.2). A copy on
both sides of a split tests the model on an image it has seen, so every
BreastMNIST result in this chapter is without them: of each group whose
labels agree, the first image is kept, and groups whose labels conflict
are dropped. Config:
[`configs/frontier_breastmnist70_nocopies.yaml`](../configs/frontier_breastmnist70_nocopies.yaml).
What the copies would change is in
[`demos/copies.ipynb`](../featsel_datasets/medmnist_breastmnist/demos/copies.ipynb),
last section. Three side studies on BreastMNIST still include them: the
classifier's C (section 3.8), RFE's C (section 7.9) and the tree count
(section 7.4); their reruns are open work (section 10.1).

## 8.9 An unseen dataset: FGVC-Aircraft

1,200 photos, Boeing 737 against the Airbus A320 family (800 against 400).
840 in the training set, 360 in the test set. Every selector ran on the
GPU, except tree importance, which has no GPU path. Berk-Jones, tree
importance, the chains and RFECV scored by accuracy were added after the
first run, on the same splits.

![Frontier on FGVC-Aircraft](figures/frontier_aircraft70_roc_auc.png)

| Method | Test AUC | Features kept | Against all features: difference (p) |
|---|---|---|---|
| All features | **0.862** | 2,048 | — |
| RFE | **0.862** | 500 to 1,000 | −0.000 (0.92) |
| RFECV, by AUC | 0.858 | 620 to 2,048 | −0.004 (0.36) |
| Mann-Whitney | 0.854 | 500 to 1,000 | −0.008 (0.34) |
| RFECV, by accuracy | 0.854 | 212 to 1,844 | −0.008 (0.37) |
| Tree importance | 0.851 | 1,000 | −0.011 (0.11) |
| Mutual information | 0.847 | 500 to 1,000 | −0.015 (0.13) |
| Lasso | 0.846 | 132 to 336 | −0.016 (0.03) |
| Mann-Whitney 500, then RFE | 0.829 | 50 to 250 | −0.033 (0.01) |
| Mann-Whitney 500, then Lasso | 0.828 | 91 to 266 | −0.034 (0.004) |
| Correlation | 0.824 | 250 to 465 | −0.038 (0.009) |
| HC2008 | 0.817 | 204 | −0.045 (< 0.001) |
| Berk-Jones | 0.815 | 183 to 204 | −0.047 (< 0.001) |
| Mann-Whitney 500, then tree importance | 0.814 | 250 | −0.049 (< 0.001) |
| Sequential, `k` = 10 | 0.703 | 10 | — |
| HC2004 | 0.674 | 1 to 2 | −0.188 (< 0.001) |

Tables:
[`figures/summary_aircraft70_chosen.csv`](figures/summary_aircraft70_chosen.csv),
[`figures/summary_aircraft70_tests.csv`](figures/summary_aircraft70_tests.csv),
[`figures/paired_tests_aircraft70.csv`](figures/paired_tests_aircraft70.csv).

The pattern of BreastMNIST holds:

- All features score best; RFE ties them.
- At small `k`, selection beats random clearly: at `k` = 10, tree
  importance by 0.127, Mann-Whitney and correlation by 0.124, RFE by
  0.103 and mutual information by 0.102, on all 16 splits
  ([`figures/paired_tests_aircraft70.csv`](figures/paired_tests_aircraft70.csv)).
- From `k` = 250, RFE leads the filters (by 0.029 over Mann-Whitney at
  250, p = 0.047, and by 0.021 at 500, p = 0.041). On BreastMNIST the
  same lead is not established (section 8.2). At `k` = 10, Mann-Whitney
  leads RFE by 0.021, not established (p = 0.43).
- Correlation's redundancy filter costs AUC: it kept at most 465 features
  and lost 0.038 (p = 0.009). On BreastMNIST it lost 0.028, not
  established (p = 0.07).
- HC2008 stops at the edge of its search, and Berk-Jones at or just before
  it; HC2004 keeps one or two features.
- Every chain loses to all features (p ≤ 0.013) and scores below its
  second method run alone, as on BreastMNIST.
- RFECV scores a little higher with AUC as its inner score than with
  accuracy (0.858 against 0.854), not established (p = 0.61).

## 8.10 Side scores and memory

Macro-F1 and balanced accuracy at the `k` each method chose, with a
threshold of 0.5 (section 3.5), on both image datasets:

| Method | BreastMNIST macro-F1 [Q1, Q3] | Balanced accuracy | FGVC-Aircraft macro-F1 [Q1, Q3] | Balanced accuracy |
|---|---|---|---|---|
| All features | 0.816 [0.792, 0.835] | 0.809 | **0.794** [0.788, 0.811] | 0.788 |
| RFE | **0.818** [0.803, 0.838] | 0.813 | 0.793 [0.783, 0.809] | 0.788 |
| Mann-Whitney | 0.810 [0.794, 0.825] | 0.804 | 0.783 [0.777, 0.799] | 0.779 |
| Mutual information | 0.812 [0.798, 0.833] | 0.806 | 0.786 [0.776, 0.797] | 0.781 |
| Tree importance | 0.806 [0.791, 0.826] | 0.799 | 0.781 [0.774, 0.793] | 0.776 |
| Correlation (Spearman) | 0.796 [0.772, 0.814] | 0.791 | 0.755 [0.739, 0.773] | 0.753 |
| RFECV, scored by AUC | 0.802 [0.788, 0.819] | 0.798 | 0.789 [0.779, 0.802] | 0.785 |
| Lasso | 0.799 [0.787, 0.817] | 0.795 | 0.782 [0.765, 0.796] | 0.777 |
| HC2008 | 0.773 [0.751, 0.796] | 0.772 | 0.741 [0.732, 0.757] | 0.738 |
| Berk-Jones | 0.770 [0.751, 0.794] | 0.769 | 0.743 [0.732, 0.763] | 0.741 |
| Sequential, `k` = 10 | not yet run | — | 0.620 [0.601, 0.638] | 0.617 |
| HC2004 | 0.663 [0.642, 0.678] | 0.651 | 0.553 [0.542, 0.563] | 0.564 |

The chains and RFECV scored by accuracy are in the tables only:
[`figures/summary_breastmnist70_nocopies_side.csv`](figures/summary_breastmnist70_nocopies_side.csv),
[`figures/summary_aircraft70_side.csv`](figures/summary_aircraft70_side.csv),
from [`scripts/summarize_side_metrics.py`](../scripts/summarize_side_metrics.py).

The picture of AUC holds. RFE ties all features. Mann-Whitney and mutual
information come within 0.011 of them. The methods that choose a smaller
count of their own (Lasso, HC, Berk-Jones) lose more. No selected set beats all
features by more than 0.003 macro-F1.

**Memory.** By the measure of section 7.12, every fit took under 20 MB,
in the median over splits. The largest single fit took 61 MB. With 2,048
features, memory does not limit any method here.

## 8.11 Simulated rare signal

**The problem.** Higher Criticism and Berk-Jones were built for signal
that is rare, weak and independent. None of the three datasets has it, so
both stopped at the edge of their search (sections 7.5, 8.5, 8.9). That
does not show whether they choose a good count where their assumptions
hold.

**The data.** We simulated such signal
([`featsel_datasets/rare_signal/`](../featsel_datasets/rare_signal/)):

- 500 samples, 150 of them positive.
- 10,000 features, each independent and standard normal.
- In 40 features, at random positions, the positive class has its mean
  moved by 0.4 standard deviations.

The signal is weak. In a training set of 350 samples, a t-test on one
informative feature gives z ≈ 3.4 on average. A Bonferroni test at 5%
needs about 4.6 at 10,000 features.

The 40 features and the shift of 0.4 come from the rare, weak setting of
the Higher Criticism tutorial
([`tutorial.ipynb`](../featsel/selectors/higher_criticism/demos/tutorial.ipynb)),
with 10,000 features in place of 2,000, so the signal is rarer. The 500
samples and the 30% positive share were set by hand.

**The protocol** is that of chapter 3: 16 splits, 70% for training, `k`
chosen on validation sets from 10, 20, 40, 80 and 160 (Mann-Whitney also
500; random also all 10,000). The runner also records the share of the 40
features each selection found (recall) and the share of the selection that
is informative (precision). One method appears only here: FDR, the
F-test with Benjamini-Hochberg control at 5% (section 4.2), the usual
filter that chooses its own count. Here the features are normal, so the
F-test's assumption holds (section 4.3). Config:
[`configs/rare_signal70.yaml`](../configs/rare_signal70.yaml). The run took
5 minutes.

| Method | Test AUC | Features kept, median (range) | Recall | Precision | HC2008 minus the method (p) |
|---|---|---|---|---|---|
| HC2008 | **0.858** | 33.5 (21 to 103) | 0.55 | 0.59 | — |
| Mann-Whitney | 0.843 | 20 (10 to 80) | 0.43 | 0.73 | 0.015 (0.62) |
| Berk-Jones | 0.813 | 12 (4 to 33) | 0.28 | 0.86 | 0.045 (0.26) |
| FDR, F-test, 5% | 0.808 | 8 (5 to 18) | 0.21 | 0.95 | 0.050 (0.22) |
| Lasso | 0.805 | 128.5 (108 to 141) | 0.72 | 0.23 | 0.053 (0.10) |
| HC2004 | 0.656 | 1.5 (1 to 9) | 0.05 | 0.99 | 0.202 (0.007) |
| All features | 0.629 | 10,000 | 1 | 0.004 | 0.229 (< 0.001) |

Tables:
[`figures/summary_rare_signal70_chosen.csv`](figures/summary_rare_signal70_chosen.csv),
[`figures/summary_rare_signal70_curve.csv`](figures/summary_rare_signal70_curve.csv),
[`figures/paired_tests_rare_signal70.csv`](figures/paired_tests_rare_signal70.csv)
and, for recall and precision, the run's own
[`results/rare_signal70.csv`](../results/rare_signal70.csv),
from [`scripts/summarize_frontier.py`](../scripts/summarize_frontier.py) and
[`scripts/rare_signal_tests.py`](../scripts/rare_signal_tests.py).

What the run shows:

- **HC2008 chooses a good count here.** It kept a median of 33.5 features
  for 40 informative ones, far below the edge of its search (1,000). It
  has the highest mean AUC, with no `k` to choose.
- **No gain over Mann-Whitney with a validated `k` is established.**
  The difference is 0.015 (p = 0.62). Nor are those to
  Berk-Jones, FDR and Lasso. HC2008 beats only HC2004 and all features.
- **The best fixed count is near the true one.** Mann-Whitney reaches
  0.864 at `k` = 20 and 0.862 at 40, then falls: 0.810 at 80, 0.695 at
  500. HC2008 finds this region without validation folds.
- **The thresholds differ in strictness.** Berk-Jones and FDR keep fewer
  features, with high precision and low recall. HC2004 keeps 1 to 9
  (median 1.5). Lasso keeps the most and finds the most, at low precision.
- **Here selection is needed.** All features give 0.629: 9,960 null
  features drown the signal. On the image data, all features were the
  best; on SCAN-B, at most 0.008 below the best in the mean over tasks,
  though 0.023 below on LumB (sections 7.1, 8.1 and 8.9).

**Limits.** One setting: 40 of 10,000 features, one shift, one simulated
dataset. The features are independent and normal, which real data are
not.
