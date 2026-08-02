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
| Fold | stratified 5-fold, fixed seed | Error bars. |

`k` is swept rather than fixed because a single `k` cannot distinguish a method
that is good at small budgets from one that only works when given many
features. Matched `k` across methods is what makes the random control valid.

### Open decisions

- **Repeats.** Currently one 5-fold split, so error bars are fold noise only,
  not split noise. Repeated stratified CV would add split noise at 3-5x the
  runtime. Undecided.
- **Number of folds.** 5 was chosen for runtime. Kohavi (1995) recommends
  10-fold stratified for lower bias. Because the study *ranks* methods rather
  than estimating absolute error, any `k` works as long as it is identical for
  every method - but if the absolute numbers are to be quoted, 10-fold is the
  defensible choice.
- **Per-class selection.** Selecting genes one-vs-rest and taking the union is
  a method variant worth testing, since a 5-class F statistic is dominated by
  the large LumA class. Not yet implemented.

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

- The selector is fit on the training split only, inside the CV loop.
  Selecting on the full dataset before splitting produced near-perfect and
  entirely spurious error rates on microarray data in Ambroise and McLachlan
  (PNAS 2002).
- Imputation and scaling are also fit inside the fold.
- PAM50 stays five classes in the multiclass framing.
- The random control is reported in the same table as every other method.
- Fixed seeds. Stochastic selectors take a **per-fold** seed derived from the
  global one: a fixed seed across folds would make their measured stability a
  meaningless 1.0.

## 5. What the current results show

From the first full grid (random and anova_f only, `results/`):

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
