# Experiment design

What the comparison varies, what it measures, and why each choice is
defensible. `SCOPE.md` decides what may be built; this file decides how the
experiment is run. Anything here that fails the four-task test in `SCOPE.md`
belongs in `docs/ideas.md` instead.

Written 2026-08-02.

## 1. The claim the experiment supports

For a given classifier and a given feature budget `k`, which feature-selection
method is best, and by how much over selecting blindly?

The comparison ranks methods under one identical protocol. It does not attempt
to produce a state-of-the-art PAM50 classifier, so absolute accuracy is
secondary to the ordering between methods and to the gap over the control.

## 2. Independent variables

| Axis | Values | What it answers |
|---|---|---|
| Selector | random (control), anova_f, lasso, tree_importance, plus wrapper and Higher Criticism when implemented | The primary question. |
| `k` | A percent of the total sample count, up to 100%, plus `k=50` and all features | How much of the gain survives as the budget grows. `k=50` is an anchor: PAM50 is a 50-gene signature. All features is the no-selection reference. See section 2d. |
| Classifier | logistic_regression, linear_svm, random_forest, knn, lda_shrinkage, xgboost | How much external selection is worth to a model, given its own regularization. |
| Task framing | 5-class PAM50, plus five one-vs-rest binary tasks | Whether the best selector differs per subtype; the 5-class number hides this. |
| Dataset | SCAN-B, plus one non-gene-expression set (task 4, undecided) | Whether conclusions generalize beyond genomics. |
| Split | 100 stratified Monte Carlo splits, fixed seed | Error bars. See section 2a. |
| Train fraction | 0.5, 0.7, 0.9 | Learning curve: does the winning method change when the training set shrinks? |

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

### Open decisions

- **Per-class selection.** Selecting genes one-vs-rest and taking the union is
  a method variant worth testing, since a 5-class F statistic is dominated by
  the large LumA class. Not yet implemented.

## 2a. Monte Carlo splits instead of k-fold

**Decision.** Replace stratified 5-fold with `StratifiedShuffleSplit`, 100
splits, at train fractions 0.5, 0.7 and 0.9. Report the **median** across
splits with an interquartile range, not the mean with a standard deviation.

**Why not k-fold.** k-fold is a structured special case of splitting, not a
free sample of it. Its test sets are forced to be disjoint, so any two training
sets in 5-fold share three quarters of their samples and the fold scores are
strongly correlated. The spread across five folds therefore understates the
real split-to-split variation, and Bengio and Grandvalet (2004) proved there is
no unbiased estimator of the variance of k-fold cross-validation. Five or ten
numbers are in any case too few to characterise a distribution.

Monte Carlo splits do not escape dependence entirely - every split is drawn
from the same 3069 patients, so training sets still overlap - but the
dependence is no longer forced by a disjointness constraint, and 100 draws
give a distribution rather than a handful of points.

**Why the median.** With 100 draws the median is insensitive to the occasional
degenerate split, and its interquartile range describes the bulk of the
distribution without assuming symmetry.

**What the interval does and does not mean.** The interval measures *split*
variance: how much the number moves if the same 3069 patients are re-split. It
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

Decided 2026-09-26.

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
  (Kohavi 1995). It costs 3069 refits per cell.
- Fixed C with no tuning: valid for ranking selectors, since every selector
  gets the same model, but C=1 may be poor at large `k`.

**Cost.** About 15 times the model fitting for 5 values of C and 3 inner
splits.

## 2c. What each run stores

Decided 2026-09-26.

**Problem.** Every metric computed during the run is fixed at run time. A new
metric means a new run.

**Decision.** Store, for every test patient in every cell, the true label, the
final model's score, and the threshold that turns the score into a prediction.
For a binary task that is one score per patient; for the five-class task, one
per class. Every metric is derived afterwards from these.

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
fractions. `k=50` is added explicitly, since it is not a round percent of 3069.

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

## 3. Dependent variables

Predictive quality, all reported per cell:

- **macro-F1** - the headline. Accuracy is reported too but is misleading
  alone: LumA is 1540 samples and Normal is 202, so accuracy rewards ignoring
  the rare subtypes.
- **balanced accuracy** and **MCC** - imbalance-robust cross-checks.
- **G-mean** (geometric mean of per-class recalls, Kubat and Matwin 1997) -
  planned. Stricter than macro-F1 because it collapses to zero if any subtype
  is never predicted.

Selection quality:

- **Stability** - Kuncheva's (2007) consistency index over folds,
  chance-corrected so a random selector scores 0 rather than the positive
  overlap it gets by luck. A method that scores well but is unstable produces
  a gene list that is an artefact of the split.
- **Selected feature count** - only interesting for methods that choose their
  own count (Lasso without `n_features`, Higher Criticism).
- **Overlap with the published PAM50 gene list** - planned. A partial ground
  truth for "did it find the right genes", not just "did it predict well".

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
- PAM50 stays five classes in the multiclass framing.
- The random control is reported in the same table as every other method.
- Fixed seeds. Stochastic selectors take a **per-split** seed derived from the
  global one: a fixed seed across splits would make their measured stability a
  meaningless 1.0.

## 5. What the current results show

From `configs/experiment_classifiers.yaml` (random and anova_f only, 12
Monte Carlo splits at train fraction 0.5, medians over splits):

- ANOVA F beats random at every `k` and every classifier.
- The gap shrinks from 0.21-0.27 macro-F1 at `k=10` to 0.01-0.14 at `k=1000`,
  because 1000 of 9259 correlated genes already proxies most of the signal.
  The small `k` end of the curve is the informative one.
- The gain at `k=1000` orders the classifiers by how weak their internal
  regularization is: knn 0.14, random_forest 0.07, linear_svm and
  logistic_regression 0.04, lda_shrinkage 0.01.
- ANOVA F's selection stability is ~0.90 across splits; random sits at 0 as
  it must.
- Cost: linear_svm took 94% of all model-fitting time (about 24 CPU-minutes
  per split). It is what makes 100 splits expensive, not the selectors.

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
