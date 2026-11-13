# 3. How to measure a selection method

[Back to contents](README.md) · Previous: [2. The data](02_the_data.md) · Next: [4. The methods](04_the_methods.md)

A selection method cannot be judged by looking at the genes it picks. We
judge it by what those genes do for a classifier on patients it has not
seen. This chapter explains how to do that without fooling ourselves.

## 3.1 The basic flow

For one experiment, called a *cell*:

1. Split the patients into a training set and a test set.
2. Fit the selection method on the training set only. It sees every gene,
   but only the training patients. It returns `k` genes.
3. Train a classifier on the training patients, using only those genes.
4. Score the classifier on the test patients.

Repeat over many random splits, and summarize the scores.

The score judges the whole procedure (select, tune, then train), not one
gene list or one model. Each split may choose a different list.

The selector sees every gene. It sees none of the test patients' values.

**Inside each split, on the training set only**, the runner:

1. fills missing values with the training median, and standardizes every
   feature with the training mean and standard deviation;
2. fits the selector, which returns the features;
3. in the comparison, tunes the classifier's C on validation sets
   (section 3.8);
4. refits the classifier on the whole training set with the chosen C.

The test set is then scored once. Nothing about it reaches the choice of
features or of C.

## 3.2 The most common mistake: selecting on all the data

It is tempting to select genes once on all patients, and then
cross-validate only the classifier. This is wrong, and the error is large.

We showed it on pure noise: 100 patients, 5,000 random genes, random
labels, 20 genes selected, scored by 5-fold cross-validation. The true AUC
is 0.5, since there is nothing to find.

| Where the 20 genes were chosen | Test AUC |
|---|---|
| On all patients, before splitting | 0.96 |
| On the training patients only | 0.44 |
| The truth | 0.50 |

Selecting on all patients picks genes that happen to fit the test patients
by chance. The test then rewards that chance. Ambroise and McLachlan
(2002) showed the same on real microarray data.

The same happens on BreastMNIST with shuffled labels. The F-test picks
50 embedding dimensions; the score is the median AUC over 20 splits.
Selected inside each split, the AUC is 0.465. Selected on all 780 images
first, it is 0.645
([`selection.ipynb`](../featsel_datasets/medmnist_breastmnist/demos/selection.ipynb)).

In `featsel` the selector is always fit on the training set of the split.
So are imputation and scaling.

## 3.3 Random splits instead of k-fold

We use stratified Monte Carlo splits: each split draws a fresh random
training set, with the class proportions kept. We do not use k-fold
cross-validation.

- In 5-fold, any two training sets share three quarters of their samples.
  So the fold scores are strongly correlated, and their spread understates
  the real variation. Bengio and Grandvalet (2004) proved that no unbiased
  estimator of the variance of k-fold cross-validation is valid for every
  data distribution.
- Five numbers are too few to describe a distribution. Random splits can be
  drawn as many times as time allows.
- Leave-one-out was rejected, for three reasons. Each test set is one
  patient, so no AUC or macro-F1 can be computed per split; only one pooled
  number remains, with no spread. Any two training sets differ by one
  patient, so stability reads near 1.0 for every method. And it costs one
  model per patient: 2,976 per cell on SCAN-B.

Random splits do not remove the dependence entirely. Every split is drawn
from the same 2,976 patients, so training sets still overlap. But the
dependence is no longer forced by k-fold's rule that the test sets be
disjoint.

**On SCAN-B, one split serves all five tasks.** Each split is stratified on
the five-class PAM50 label. So every subtype-vs-rest task is scored on the
same patients.

**How many splits.** The first plan was 100 splits at three training
fractions (0.5, 0.7 and 0.9), to draw a learning curve. The full grid would
have taken about four days. The comparison ran 16 splits on SCAN-B and 20
on BreastMNIST, with half the patients for training. Sixteen splits keep
the SCAN-B run under half an hour. Sixteen or more values support
quartiles; five do not.

**More splits help less than they seem.** Random splits share most of
their training samples. So the corrected test of section 7.3 multiplies
the variance by 1/J + n_test/n_train, for J splits. The second term does
not shrink as J grows. So the 95% interval cannot become narrower than a
fixed limit, however many splits are run. The table gives the interval's
width as a multiple of that limit:

| Splits J | 5 | 10 | 16 | 20 | 32 | 100 |
|---|---|---|---|---|---|---|
| Half for training | 1.55 | 1.21 | 1.12 | 1.09 | 1.06 | 1.02 |
| 70% for training | 1.72 | 1.28 | 1.16 | 1.13 | 1.08 | 1.02 |

The width combines the variance factor with the t value for J − 1
degrees of freedom. Going from 5 splits to 16 narrows the interval by 28
to 32%. Going beyond 16 can narrow it by at most 14% more. So 16 is not
only what the laptop allowed. On any machine, it gives most of what extra
splits can give. A difference the test cannot establish at 16 splits will
rarely be established by more splits. It needs more patients instead.

We report the **median over splits with the first and third quartiles**,
or the mean with the standard deviation where noted. The median ignores
the occasional odd split. The quartiles describe the bulk of the values
without assuming they are symmetric.

**What the spread means.** The spread measures how much a score moves when
the same patients are split again. It is not a confidence interval for new
patients, because every split reuses the same cohort.

**Paired comparison.** All methods run on the same splits. So two methods
can be compared split by split. This removes most of the split-to-split
noise. A method's gain over the random control is the method's score minus
the control's score, on the same split and at the same `k`.

**Testing a paired difference.** The splits overlap, so their differences
are not independent. A standard paired test would overstate certainty.
Chapter 7 uses the corrected resampled t-test of Nadeau and Bengio
(2003), which widens the variance to allow for the overlap (section 7.3).

## 3.4 The random control

The control picks `k` features at random, with a new draw on every split.
It gets the same splits, the same `k` and the same classifier as every
method.

A comparison without it cannot tell whether a method finds signal, or only
reduces the number of features. Neekhra et al. (2025) found that small
random subsets matched or beat feature selection on 28 of 30
high-dimensional datasets. As we show in chapter 7, this happens on our
data too once `k` is large.

## 3.5 Scores

The tasks are imbalanced (chapter 2), so plain accuracy is misleading.

| Score | What it measures | Why we use it |
|---|---|---|
| **ROC AUC** | The chance that the model scores a random positive patient above a random negative one. | It needs no decision threshold. So it does not mix the method's quality with a choice of costs. Main score of the frontier (chapter 8). |
| **Macro-F1** | The average of the F1 score of each class. | It weights the small class as much as the large one. Main score of the comparison (chapter 7), because the runner began recording test AUC only after the comparison ran. |
| **Balanced accuracy** | The average of the recall of each class. | A simple imbalance-robust check. |

The comparison also records:

- **MCC**, the Matthews correlation coefficient, another imbalance-robust
  check.
- **G-mean**, the geometric mean of the two class recalls. It is stricter
  than macro-F1: it is zero if either class is never predicted correctly.
- **Accuracy**, for comparison with other work. Alone it misleads. On
  Normal vs rest (200 against 2,776 patients), always predicting "rest"
  scores 0.93.

**Why not F1 alone.** F1 fixes a trade-off between false positives and
false negatives. That is a decision about costs, which a comparison of
selection methods should not make. Accuracy, balanced accuracy, MCC and
G-mean also depend on a decision threshold. So each implies some costs
too. AUC needs neither a threshold nor costs.

AUC is computed per split, from one model on that split's test patients.
Pooling scores from different models into one AUC would compare numbers
that are not on the same scale.

## 3.6 Stability

A method can score well and still pick different genes on every split.
Its gene list is then an accident of the split, not a finding.

We measure stability by Kuncheva's (2007) consistency index. It compares
the gene sets of two splits. We average it over all pairs of splits. It is 1 for
identical sets and 0 for the overlap expected by chance. So the random
control scores about 0. A pairwise measure needs many splits: 5 splits
give 10 pairs, 16 give 120, and 20 give 190.

Kuncheva's index assumes that both sets have the same size. Some methods
choose their own count, so their sets differ in size from split to split.
For them, our index uses the mean size of each pair, which is an
approximation. So we also compute the measure of Nogueira, Sechidis and
Brown (2018). It needs no equal sizes. It is built from how often each
feature is selected across splits. When the sizes are equal, it equals the
mean pairwise Kuncheva index.

A random method must get a different seed on each split. With the same
seed on every split, its measured stability would be a meaningless 1.0.

## 3.7 Cost

For every fit we record:

- **Selection time**: the time of the selector's `fit` alone, not of the
  classifier. In the frontier each selection is fit 10 times, each time by
  a fresh selector, and the median time is recorded. Sequential selection
  takes minutes per fit, so it is fit once.
- **Search time**: the time to choose `k`, or a method's own
  hyperparameter, on validation sets (section 3.11). It is recorded apart
  from the selection time.
- **Peak memory** of the selection.
- **The device**: CPU or GPU.
- **Reproducibility**: whether every repeated fit chose the same genes.

Classifier training time is not part of the cost. It is the same for every
method at the same `k`.

The design also planned to state each method's time and memory complexity,
derived from its algorithm, next to the measurements. A disagreement
between the two would itself be worth reporting. This was not done.

## 3.8 Tuning the classifier without leakage

The classifier, L2-penalized logistic regression, has a strength parameter
C. In the comparison (chapter 7) C is tuned inside each split, in three
steps:

1. The training set is split at random three times. Each time, half of it
   is used to fit the classifier and the other half is a validation set.
2. Each C from {0.01, 0.1, 1, 10, 100} is scored by its AUC on the three
   validation sets.
3. The classifier is refit on the whole training set with the best C.

The test set is used once, at the end.

The penalty is (λ/2)‖w‖², and scikit-learn's C is 1/λ. Features are
standardized on the training set first. Otherwise the penalty would treat
features on different scales differently. scikit-learn's default solver,
`lbfgs`, does not penalize the bias term. (The `liblinear` solver does.)

C is scored by AUC, not macro-F1. AUC does not depend on a decision
threshold, so it scores C more smoothly on small validation sets.

Tuning costs about 15 times the classifier's fitting: 5 values of C times
3 validation sets.

The selector is fit once on the whole training set, not once per
validation set. So the validation patients helped choose the genes. This
can bias the choice of C slightly. The test set stays untouched, so the
reported scores stay honest.

In the frontier (chapter 8) the classifier's C is fixed at 1, to keep the
run short. Every method gets the same classifier, so the ranking stays
fair. But the absolute scores may be a little lower than with tuning. This
C is the classifier's. It is not Lasso's own C, which the frontier chooses
on validation sets (section 3.11).

## 3.9 Reproducibility

There is one global seed, 42. Split `s` uses seed 42 + `s` for every
random step in it: the random selector, the forest, the solvers and the
validation splits. Every selector that uses randomness seeds its own random
generators in its `fit`. Re-running a configuration gives the same genes
and the same scores. Only the times change. The number of workers also
changes only the times.

## 3.10 Preprocessing is fixed

Every run uses **standardization only**, fit on the training set. Adding
preprocessing as another axis of the experiment would double or quadruple
an already large grid. And it would answer a question that is not one of
the project's four tasks.

**Min-max scaling was rejected without a test.** Like standardization, it
shifts and scales each feature. So it changes nothing for methods that
ignore a feature's scale, such as the F-test, correlation and trees. It
differs only for methods that depend on scale, such as k-NN and penalized
models (L1 or L2). There it is worse: one outlier sets the range and
squeezes every other sample.

**A side experiment.** Each feature could be mapped to a normal
distribution through its ranks (a quantile transform). 98.1% of genes
reject normality, so there is something to fix. But the transform
discards the size of a fold change.

We tested it on `scanb_small`, with F-test selection, logistic regression
and 10 splits. Standardization was compared with a rank-to-normal
transform and with scikit-learn's quantile transform. The two transforms
tied with each other.

- On five classes, standardization won at every `k`. Macro-F1 was 0.734
  against 0.597 at `k` = 10, and 0.870 against 0.855 at `k` = 500.
- On the five binary tasks, most differences were within 0.01. Her2 and
  Normal gained up to 0.011. LumA lost up to 0.021; at `k` = 100 it lost
  on all 10 splits.
- With C tuned on validation sets and scored by AUC, differences stayed
  within 0.004, except LumA at `k` = 100 (−0.017).

So standardization was kept. Two classifiers might still gain from the
transform: shrinkage LDA, which assumes normal classes, and k-NN, which
suffers from heavy tails. A run over every classifier was started and
cancelled, so this is unknown.

**One rule.** If preprocessing is ever varied, it must be varied for every
method alike. A method that carries its own preprocessing mixes the method
with its transform. Then the comparison is no longer controlled.

## 3.11 The feature budget `k`

In the comparison, `k` on SCAN-B is 5, 10, 20, 50, 100, 250, 500 and 1000.
On BreastMNIST it stops at 250.

- **Why a sweep.** A single `k` cannot tell a method that is good at small
  budgets from one that works only when given many features.
- **Why the same `k` for every method.** It makes the comparison with the
  random control fair.
- **Why 50.** PAM50 is a 50-gene signature, so `k` = 50 is an anchor.
- **Why the largest `k` stays below the training size.** The point of
  selection here is to get below the number of samples. Selecting more
  features than training samples is not a useful target.
- **Selection against no selection.** A run on all features is kept as a
  reference. It shows whether selection helps the model or only makes it
  smaller ([section 8.7](08_results_frontier.md#87-is-selection-needed-at-all)).

The first design gave `k` as a percent of the total sample count. Budgets
would then compare across datasets and stay fixed across training fractions.
With one training fraction, the experiments give `k` directly.

**Choosing `k` on validation sets.** The goal is the best AUC, so a method
should choose its own `k`. Choosing it by test AUC would use the test set
twice: once to choose and once to score. So in the frontier, each method
with a `k` list chooses `k` on validation sets:

1. The training set is cut into 5 folds. Each fold in turn is the
   validation set.
2. The selector is fit on the other four folds.
3. For each `k`, logistic regression is fit on the top `k` features and
   scored by AUC on the validation set.
4. The `k` with the best mean validation AUC wins. A tie goes to the
   smaller `k`.
5. That `k`'s classifier, fit on the whole training set, is scored once on
   the test set.

A method that ranks once fits once per fold, and cuts every `k` from that
fit. So the search costs 5 extra fits, against the 10 timed fits of each
selection. Its time is recorded apart from the selection time.

Standardization is fit on the whole training set, so the validation sets
helped set its means and spreads. That uses no labels. The test set stays
untouched.

On SCAN-B the frontier's `k` are 10, 50, 250, 500, 1000, 1500 and 2000,
the same for every method. The first frontier run stopped at 1,000. There,
AUC was still rising for every method with a `k` list.

Sequential selection does not search, because one of its fits takes
minutes.

**A searching method is timed once.** Lasso chooses its C from four values
by mean AUC over 5 validation sets. RFECV chooses its count by 5-fold
cross-validation on the training set. Each searches once. Its 10 timed
repeats then fit with what the search chose: Lasso with that C, RFECV as
RFE to that count. The search's time is recorded on its own. The repeats
select the same features as the search, and the runner checks this.

## 3.12 Data precision

The data are held as 32-bit floats. That halves the data's memory. Peak
selection memory on SCAN-B fell by 30% to 45%, so more workers fit on one
machine. SCAN-B's values are not measured to more than 7 digits, so 32
bits lose nothing measured.

The U test sums its ranks in 64 bits. When it summed them in 32 bits, the
tie correction of near-constant genes went wrong. Mann-Whitney then chose
different genes in every SCAN-B cell, and macro-F1 moved by up to 0.55.
After the fix, a one-split check against 64-bit data moved Mann-Whitney's
macro-F1 by at most 0.007. Lasso, RFE and the forest moved by about 0.01
on average, because their solvers are sensitive to rounding.

## 3.13 Scores we considered and did not use

We considered two ways to give each method a single number, and chose
neither.

- **A leave-one-out confusion table.** It leaves out one patient at a time
  and counts the decisions. It trains on almost all patients. But it
  brings back the decision threshold, and it gives one table with no
  spread.
- **The area under a score-against-`k` curve,** minus the random
  control's area. It depends on how the `k` axis is scaled. And methods
  that pick their own count give a point, not a curve.

We report the whole curves instead. One number hides where along `k` a
method wins.

## 3.14 Limits of this protocol

- The spread over splits is not a confidence interval for new patients.
- The five SCAN-B tasks share patients, so they are not independent
  replicates.
- The decision threshold is not tuned: 0.5 on the predicted probability.
  Macro-F1 and balanced accuracy depend on it. AUC does not.
- In the comparison, the selectors' own settings, such as Lasso's penalty,
  are fixed. In the frontier, Lasso chooses its penalty on validation sets
  (section 3.11). The other settings stay fixed there too.
- Choosing a "best `k`" from test scores would leak. So the comparison
  reports whole curves over `k` and names no winning `k`. The frontier
  chooses `k` on validation sets instead (section 3.11).
- The runner stores each cell's metrics, not each test patient's score. So
  a new metric needs a new run. Storing the scores was designed but not
  built (chapter 10).
