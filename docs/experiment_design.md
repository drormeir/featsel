# Experiment design

What the comparison varies, what it measures, and why each choice is
defensible. `SCOPE.md` decides what may be built; this file decides how the
experiment is run. Anything here that fails the four-task test in `SCOPE.md`
belongs in `docs/ideas.md` instead.

Written 2026-08-02. Revised 2026-10-31 to the protocol the runs use: five
splits at one train fraction, the U test for every filter, tuned C.

## 1. The claim the experiment supports

For a given classifier and a given feature budget `k`, which feature-selection
method is best, and by how much over selecting blindly?

The comparison ranks methods under one identical protocol. It does not attempt
to produce a state-of-the-art PAM50 classifier, so absolute accuracy is
secondary to the ordering between methods and to the gap over the control.

## 2. Independent variables

| Axis | Values | What it answers |
|---|---|---|
| Selector | RandomSelector (control), ANOVAFSelector, LassoSelector, TreeImportanceSelector, RFESelector, HigherCriticismSelector in its 2004 and 2008 variants. Filters score genes by the Mann-Whitney U test (section 2f). | The primary question. |
| `k` | Selector comparison: 10, 50 and 250. Classifier study: 10 to 1000. Selectors that choose their own count (Higher Criticism) take no `k`. `k=50` is an anchor: PAM50 is a 50-gene signature. See section 2d. | How much of the gain survives as the budget grows. |
| Classifier | Selector comparison: logistic_regression. Classifier study (`notebooks/02_classifier_behaviour.ipynb`): logistic_regression, linear_svm, random_forest, knn, lda_shrinkage. | How much external selection is worth to a model, given its own regularization. |
| Task framing | One binary task per class, that class vs the rest: five for PAM50. No five-class task (decided 2026-10-18, `SCOPE.md` 3a). | Whether the best selector differs per subtype. |
| Dataset | SCAN-B, and BreastMNIST ResNet-50 embeddings (task 4, decided 2026-10-17) | Whether conclusions generalize beyond genomics. |
| Split | Selector comparison: 5 stratified Monte Carlo splits. Classifier study: 12. Fixed seed. | Error bars. See section 2a. |
| Train fraction | 0.5 | Fixed. The learning curve over 0.5, 0.7 and 0.9 was dropped for time (section 2a). |

`k` is swept rather than fixed because a single `k` cannot distinguish a method
that is good at small budgets from one that only works when given many
features. Matched `k` across methods is what makes the random control valid.

### Preprocessing is fixed, not swept

The main grid uses **standardization only**, fit inside the split. Adding a
preprocessing axis would double or quadruple an already large grid for a
question that is not one of the four tasks.

Min-max was rejected rather than tested: it is affine per column, exactly like
standardization, so it changes nothing for anything scale-equivariant - ANOVA
F, Pearson correlation and tree ensembles give identical results either way.
It differs only for k-NN and L2-penalised models, and there it is the worse
choice, because one outlier sets the range and compresses every other sample.

**Side experiment (not in the main grid).** Gaussianizing each column with a
rank-based normal quantile transform, at one `k` and one train fraction, run
across all classifiers. The prediction is that shrinkage LDA and k-NN gain,
because the first assumes Gaussian class densities and the second is dominated
by heavy tails, while random forest and XGBoost do not move at all, being
invariant to monotone transforms. It may also *hurt*: the transform is
rank-based, so it discards the magnitude of a fold change, and its quantiles
are estimated on the training half only, so at `train_size=0.5` the mapping is
noisy and out-of-range test values are clipped. A measured check on SCAN-B
found 96.7% of columns rejecting normality (median excess kurtosis 1.15), so
there is something for it to fix. Result still unknown: the run was started and
cancelled.

If preprocessing is ever crossed with the other axes, it must be crossed for
*every* method, not attached per algorithm. A method carrying its own
preprocessing confounds the method with its transform, and the comparison stops
being controlled.

### Binary tasks only

Decided 2026-10-18. Features are selected for binary classifiers only. A
target with more than two classes becomes one dataset per class, that class
vs the rest. The datasets differ only in the target. The five PAM50 classes
are too imbalanced to report as one task.

Rejected: the five-class task, and the union of the per-class gene sets.
Recombining the binary classifiers into a subtype decision tree is parked in
`docs/ideas.md`.

## 2a. Monte Carlo splits instead of k-fold

**Decision.** Replace stratified 5-fold with `StratifiedShuffleSplit`.
Report the **median** across splits with an interquartile range, not the mean
with a standard deviation.

**Revised 2026-10-31: 5 splits at train fraction 0.5.** The original plan,
100 splits at 0.5, 0.7 and 0.9, needs about four days for the full SCAN-B
grid. The selector comparison uses 5 splits; the classifier study has 12.
Consequence: a median and quartiles over 5 values are rough, so the report
also gives the range, and Kuncheva's index rests on 10 pairs of splits. The
reasoning below, written for 100 splits, still explains why splits are drawn
at random rather than as folds.

Each split is stratified on the full target and shared by every task, so all
subtype-vs-rest tasks are scored on the same patients.

**Why not k-fold.** k-fold is a structured special case of splitting, not a
free sample of it. Its test sets are forced to be disjoint, so any two training
sets in 5-fold share three quarters of their samples and the fold scores are
strongly correlated. The spread across five folds therefore understates the
real split-to-split variation, and Bengio and Grandvalet (2004) proved there is
no unbiased estimator of the variance of k-fold cross-validation. Five or ten
numbers are in any case too few to characterise a distribution.

Monte Carlo splits do not escape dependence entirely - every split is drawn
from the same 2976 patients, so training sets still overlap - but the
dependence is no longer forced by a disjointness constraint, and 100 draws
give a distribution rather than a handful of points.

**Why the median.** With 100 draws the median is insensitive to the occasional
degenerate split, and its interquartile range describes the bulk of the
distribution without assuming symmetry.

**What the interval does and does not mean.** The interval measures *split*
variance: how much the number moves if the same 2976 patients are re-split. It
is not a confidence interval over the population of breast cancer patients,
because every split reuses the same cohort. The report must say this
explicitly.

**Cost.** 100 splits is 20x the runtime of a single 5-fold run, multiplied
again by three train fractions. This is the concrete workload that task 3
(parallel infrastructure) exists to serve, and it is the reason the grid is
worth parallelizing at all.

**Stability under this scheme.** Kuncheva's index is a pairwise measure, so 5
folds give only 10 pairs. 100 splits give 4950, which turns the stability
number from an estimate into a distribution. Pairs may be subsampled if the
count becomes the bottleneck; if so, the subsample size is reported.

## 2b. Tuning inside the split: nested Monte Carlo

Decided 2026-09-26. Confirmed 2026-10-31; being built. Until it is, the
runner uses the fixed C given in the config.

**Problem.** Regularized models have a strength parameter, such as C. Choosing
it by test scores makes the test scores optimistic.

**Decision.** Two levels of Monte Carlo splits. Outer splits estimate
performance. Inside each outer training part, inner stratified Monte Carlo
splits choose C from a small fixed grid. The model is then refit on the whole
outer training part with that C, and scored once on the outer test part.

- The selector is fit once per outer training part, not inside each inner
  split. The inner validation samples then helped choose the genes, which can
  make the chosen C slightly off. The test part stays untouched, so test scores
  stay honest.
- A decision threshold, if tuned, is chosen on the inner splits too.
- Choosing a "best k" from test scores is the same leak. Report whole curves
  over `k`. Naming a winning `k` needs inner validation.

**Rejected.**

- k-fold cross-validation, inner or outer: see section 2a.
- Leave-one-out: a one-patient test set gives no per-split macro-F1, only a
  pooled number with no spread. Training sets differ by one patient, so
  stability reads near 1.0 for every method. Its estimate has high variance
  (Kohavi 1995). It costs 2976 refits per cell.
- Fixed C with no tuning: valid for ranking selectors, since every selector
  gets the same model, but C=1 may be poor at large `k`.

**Cost.** About 15 times the model fitting for 5 values of C and 3 inner
splits.

## 2c. What each run stores

Decided 2026-09-26. Not built: the runner stores the metrics of each cell,
not the score of each patient. A new metric still needs a new run.

**Problem.** Every metric computed during the run is fixed at run time. A new
metric means a new run.

**Decision.** Store, for every test patient in every cell, the true label, the
final model's score, and the threshold that turns the score into a prediction.
Every task is binary, so that is one score per patient. Every metric is
derived afterwards from these.

- Threshold metrics (accuracy, F1, balanced accuracy, MCC, G-mean) come from
  the confusion matrix, which the scores and threshold reproduce.
- Ranking metrics (ROC-AUC, PR-AUC) need the scores themselves.
- The threshold is 0.5 on a probability or 0 on an SVM decision value, unless
  tuned as in section 2b.

**Limits.** Scores are not comparable across models: linear SVM gives raw
decision values, random forest and k-NN coarse probabilities. At 100 splits
the scores run to tens of millions of numbers, so they need a compact file
format such as Parquet, not CSV.

## 2d. The feature budget `k`

Decided 2026-09-26.

**Decision.** `k` is given as a percent of the total sample count, so the
budget scales with the data and is comparable across datasets. It uses the
total count, not the training count, so `k` stays fixed across train
fractions. `k=50` is added explicitly, since it is not a round percent of 2976.

**Above the sample count.** Selecting more features than training samples is
not a useful target: the point of selection here is to get below it. One run
with all features is kept as the reference, to show whether selection helps the
model or only makes it smaller. Logistic regression still peaked at the largest
`k` tested (1000) in section 5, so the curve's peak is not yet known.

## 2e. Speed against plain scikit-learn

Decided 2026-09-26. This is how task 3 is measured.

**Problem.** Most selectors and all models come from scikit-learn. The runner's
value over plain scikit-learn has to be shown in numbers, not claimed.

**Decision.** Run the same small grid two ways, on the same
`StratifiedShuffleSplit` splits: plain scikit-learn (`GridSearchCV` or
`cross_validate`), and the runner on 1 core and on 12. Check that the scores
are identical. Report wall-clock time, with the speedup split by source:

- parallelism over splits;
- ranking features once per split and taking every `k` from that ranking,
  where `GridSearchCV` refits the selector for each `k`;
- imputation and scaling once per split, shared by every selector and model.

A subset is enough, for example 5 splits, 2 selectors and 3 values of `k`,
extrapolated to the full design.

**Why it matters.** Extrapolated from the 12-split run, the full design (300
splits, 6 selectors, nested tuning) needs about 30-35 CPU-hours without linear
SVM: a day and a half on one core, 3-4 hours on 12. RFE removing one feature
per round would take about 2300 hours; removing 10% per round and ranking once
takes about 3. That is the proposal's point that some techniques are
impractical in this setting.

**Limits.** "Impractical", not "impossible", for most of the grid: 35 hours on
one core is slow but feasible. Linear SVM with its default dual solver costs
hundreds of hours; `dual=False` may fix that and is not yet verified.

## 2f. The univariate test: Mann-Whitney U for every filter

Decided 2026-10-31. Every filter selector scores genes by the two-sided
Mann-Whitney U test: ANOVAFSelector, FDRSelector, FWESelector,
PercentileSelector and HigherCriticismSelector.

**Why.** The gene values are log2(FPKM + 0.1), an arbitrary monotone
transform, so only the order of each gene's values is meaningful. The U test
uses only that order: U is the gene's AUC times n1 n2. The F-test depends on
the size of the values and assumes normal values within each class, while
about half the values sit tied at the floor. The full reasoning is in
`docs/report/03_data.md`, "Decided: HC scores genes by the Mann-Whitney U
test". Using one test for every filter keeps the comparison between them a
comparison of selection rules, not of tests.

**Consequence.** These selectors no longer select what scikit-learn's
F-test selectors do, so their tests compare against scipy's U test instead.
A fit costs about 10 s on SCAN-B, against about 1 s for the F-test.

## 3. Dependent variables

Predictive quality, all reported per cell:

- **macro-F1** - the headline. Accuracy is reported too but is misleading
  alone: Normal vs the rest is 202 against 2774 patients, so predicting "rest"
  for everyone scores 0.93 accuracy.
- **balanced accuracy** and **MCC** - imbalance-robust cross-checks.
- **G-mean** (geometric mean of the two class recalls, Kubat and Matwin 1997).
  Stricter than macro-F1 because it collapses to zero if either class is
  never predicted.

Selection quality:

- **Stability** - Kuncheva's (2007) consistency index over splits,
  chance-corrected so a random selector scores 0 rather than the positive
  overlap it gets by luck. A method that scores well but is unstable produces
  a gene list that is an artefact of the split.
- **Selected feature count** - only interesting for methods that choose their
  own count (Lasso without `n_features`, Higher Criticism).
- **Overlap with the published PAM50 gene list** - planned. A partial ground
  truth for "did it find the right genes", not just "did it predict well".
  PAM50 labels are computed from these 50 genes, so for the PAM50 target the
  answer is known. It needs a gene set that holds all 50: GSE96058 does;
  the retired `scanb_small` lacked 11 of them. `run(..., truth=...)`
  already records recall and precision against given feature indices. See
  `docs/report/03_data.md`.

Cost:

- **Selection time**, measured separately from classifier training time.
  Classifier training is a fixed business cost and is not what task 3
  parallelizes.
- **Peak selection memory**, via `tracemalloc` around the selector fit.
- **Analytical time and space complexity** per method, derived from the
  algorithm and reported next to the measurements. Disagreement between the
  analysis and the measurement is itself worth reporting.

## 4. Protocol rules

These come from `SCOPE.md` 3a and are not negotiable.

- The selector is fit on the training split only, inside the resampling loop.
  Selecting on the full dataset before splitting produced near-perfect and
  entirely spurious error rates on microarray data in Ambroise and McLachlan
  (PNAS 2002).
- Imputation and scaling are also fit inside the split.
- Every task is binary. A multiclass target becomes one task per class, that
  class vs the rest.
- The random control is reported in the same table as every other method.
- Fixed seeds. Stochastic selectors take a **per-split** seed derived from the
  global one: a fixed seed across splits would make their measured stability a
  meaningless 1.0.

## 5. What the current results show

From `configs/experiment_classifiers.yaml`, run 2026-10-31: RandomSelector
and ANOVAFSelector (F-test, before section 2f), five classifiers, the five
subtype-vs-rest tasks, 12 splits at train fraction 0.5, fixed C. Figures in
`notebooks/02_classifier_behaviour.ipynb`.

- ANOVA beats random by 0.20-0.24 macro-F1 at `k=10`, for every classifier.
- The gap closes as `k` grows, and for LDA reverses at `k=1000`: a large
  random subset of 28,552 correlated genes already carries the signal.
- The gain differs by subtype: at `k=10` from 0.15 (LumA) to 0.33 (Her2).
- Best: logistic regression, 0.869 at `k=1000`.

The selector comparison has only a one-split smoke test so far.

## 6. References

- Ambroise, C. and McLachlan, G. (2002). Selection bias in gene extraction on
  the basis of microarray gene-expression data. *PNAS* 99(10).
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
- Bengio, Y. and Grandvalet, Y. (2004). No unbiased estimator of the variance
  of k-fold cross-validation. *JMLR* 5.
- Beyer, K., Goldstein, J., Ramakrishnan, R. and Shaft, U. (1999). When is
  "nearest neighbor" meaningful? *ICDT*.
- Donoho, D. and Jin, J. (2008). Higher criticism thresholding. *PNAS* 105(39).
- Friedman, J. (1989). Regularized discriminant analysis. *JASA* 84(405).
- Guyon, I. and Elisseeff, A. (2003). An introduction to variable and feature
  selection. *JMLR* 3.
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy
  estimation and model selection. *IJCAI*.
- Kubat, M. and Matwin, S. (1997). Addressing the curse of imbalanced training
  sets: one-sided selection. *ICML*.
- Kuncheva, L. (2007). A stability index for feature selection. *IASTED
  Artificial Intelligence and Applications*.
- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate Analysis*
  88(2).
- Parker, J. et al. (2009). Supervised risk predictor of breast cancer based on
  intrinsic subtypes. *Journal of Clinical Oncology* 27(8).
- Saeys, Y., Inza, I. and Larranaga, P. (2007). A review of feature selection
  techniques in bioinformatics. *Bioinformatics* 23(19).
