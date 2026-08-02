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
| `k` | 10, 25, 50, 100, 250, 500, 1000 | How much of the gain survives as the budget grows. `k=50` is an anchor: PAM50 is a 50-gene signature. |
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

From the first full grid (random and anova_f only, stratified 5-fold, before
the switch to Monte Carlo splits in section 2a, `results/`):

- ANOVA F beats random at every `k` and every classifier.
- The gap shrinks from ~0.23 macro-F1 at `k=10` to ~0.05 at `k=1000`, because
  1000 of 9259 correlated genes already proxies most of the signal. The small
  `k` end of the curve is the informative one.
- The gain at large `k` orders the classifiers by how weak their internal
  regularization is: knn 0.15, random_forest 0.07, linear_svm and
  logistic_regression ~0.05-0.06, lda_shrinkage 0.01.
- ANOVA F's selection stability is ~0.95 across folds; random sits at 0 as it
  must.

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
