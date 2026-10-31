# Methodology

Draft. The protocol is decided in `docs/experiment_design.md`; this chapter
states it for the reader. Configs: `configs/comparison_scanb.yaml` and
`configs/comparison_breastmnist.yaml`.

## The question

For a given feature budget `k`, which selection method gives the best
classifier, and by how much over choosing features at random?

The comparison ranks methods under one protocol. It does not aim at the best
possible classifier. The order between methods, and the gap over the random
control, matter more than the absolute scores.

## Tasks

Every task is binary. A target with more than two classes becomes one task
per class: that class against the rest. The tasks share the patients and
differ only in the labels.

On SCAN-B the target is the PAM50 subtype, so there are five tasks: Basal,
Her2, LumA, LumB and Normal, each against the rest. They are decompositions
of one target, not independent datasets. On BreastMNIST there is one task:
malignant against benign or normal.

## Selection methods

| Method | Family | How it ranks or chooses features | Settings |
|---|---|---|---|
| Random | control | Draws `k` features at random, a new draw per split | |
| Mann-Whitney | filter | Smallest two-sided U-test p-value, one feature at a time | |
| Lasso | embedded | Largest absolute coefficient of L1-penalized logistic regression | C = 1 |
| Tree importance | embedded | Largest mean impurity decrease in a random forest | 200 trees |
| RFE | wrapper | Recursive elimination with a linear SVM, 10% of the features per round (Guyon et al. 2002) | C = 0.01 |
| HC 2004 | filter, own count | Higher Criticism threshold on U-test p-values (Donoho and Jin 2004) | alpha0 = 0.1 |
| HC 2008 | filter, own count | Higher Criticism thresholding with the exact null mean i/(N+1) (Donoho and Jin 2008) | alpha0 = 0.1 |

Every filter uses the Mann-Whitney U test. Gene values are on an arbitrary
monotone scale, log2(FPKM + 0.1), and the U test uses only their order. One
test for every filter makes the comparison between filters a comparison of
selection rules, not of tests (`docs/experiment_design.md` 2f).

The first five methods take `k`. Both Higher Criticism variants choose their
own count, so their results stand beside the fixed-`k` curves rather than
in them.

`k` is 10, 50 and 250. 50 matches the size of the PAM50 signature. 250 is
below the training size on both datasets: selection here aims at fewer
features than samples.

## Evaluation protocol

**Outer splits.** Five stratified Monte Carlo splits, each with half the
samples for training and half for testing. On SCAN-B a split is stratified
on the five-class PAM50 label and shared by all five tasks, so every task is
scored on the same patients. Random splits are used instead of k-fold
because k-fold forces the test sets apart and correlates the fold scores
(Bengio and Grandvalet 2004).

**Inside each split, on the training half only:**

1. Missing values are filled with the training median, and every feature is
   standardized with the training mean and standard deviation.
2. The selector is fit and returns the features.
3. The classifier's C is tuned (below).
4. The classifier is refit on the whole training half with the chosen C.

The test half is then scored once. Nothing about the test half reaches the
choice of features or of C. Selecting features before splitting makes error
rates on microarray data look near perfect (Ambroise and McLachlan 2002).

**Ranking once per split.** Mann-Whitney, Lasso and tree importance rank all
features in one fit, and that fit does not depend on `k`. Their top-10,
top-50 and top-250 are cut from one ranking per split. The cut equals a
separate fit at each `k`. RFE stops at `k`, so it is fit once per `k`.

## The classifier and nested tuning

The classifier is L2-penalized logistic regression. Its strength C is tuned
inside each split, from the grid 0.01, 0.1, 1, 10 and 100.

The training half is split again: three stratified Monte Carlo splits, half
for inner training and half for inner validation. Each C is scored by the
mean ROC-AUC over the three validation parts, and the best one is kept.
ROC-AUC does not depend on the decision threshold, so it scores C more
smoothly than macro-F1 on small validation parts.

The selector is fit once on the whole training half, not inside each inner
split. The inner validation samples therefore helped choose the features,
which can bias the choice of C slightly. The outer test half is untouched,
so the reported scores stay honest.

## Metrics

Each execution, one (task, split, method, `k`) cell, records:

- **macro-F1**, the headline. Each one-vs-rest task is imbalanced: Normal
  against the rest is 200 against 2,776 patients, so always predicting
  "rest" scores 0.93 accuracy.
- **Balanced accuracy, MCC and G-mean** as imbalance-robust checks. G-mean is
  zero if either class is never predicted.
- **Accuracy**, for comparison with other work.
- **Selection time** and **peak selection memory**, for the fit of the
  selector alone.
- **The number of features chosen**, which varies only for Higher Criticism.
- **The chosen C.**

Over the five splits each cell reports the median and the range. Five
values are too few for quartiles.

**Stability** is Kuncheva's (2007) consistency index between the feature sets
of two splits, averaged over the ten pairs. It is 1 for identical sets and
0 for the overlap expected by chance, so the random control scores near 0.

## Baseline

The random selector is the control. It gets the same `k`, the same splits
and the same tuned classifier as every method. A method's gain is its
macro-F1 minus the random control's, at the same `k` on the same split.

## Reproducibility

One global seed, 42. Split `s` uses seed 42 + `s` for every random step in
it: the random selector, the forest, the solvers and the inner splits.
Re-running a config gives the same features and the same scores. Only the
timings change.

## Limits

- Five splits give a rough spread. The range is reported, not a confidence
  interval. It measures how much a number moves when the same patients are
  re-split, not how it would move on new patients.
- The five SCAN-B tasks share patients, so they are not independent
  replicates.
- The decision threshold is not tuned: 0.5 on the predicted probability.
- Lasso and RFE use a fixed C inside the selector. Only the classifier's C is
  tuned.
