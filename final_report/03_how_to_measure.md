# 3. How to measure a selection method

[Back to contents](README.md) · Previous: [2. The data](02_the_data.md) · Next: [4. The methods](04_the_methods.md)

A selection method is judged by what its genes do for a classifier on
patients it has not seen. This chapter describes the one protocol used for
every experiment.

| Part | Choice | Section |
|---|---|---|
| Splits | 16 random stratified splits, 70% of samples in the training set | 3.3 |
| Main score | ROC AUC: mean over splits, quartiles for spread | 3.3, 3.5 |
| Side scores | Macro-F1, balanced accuracy | 3.5 |
| Baseline | Random features at each `k`: a method has value at `k` only above them | 3.4 |
| `k` | Chosen on validation sets: 5 folds of the training set, best mean AUC, tie to the smaller `k` | 3.11 |
| Classifier | Logistic regression, C fixed at 1 | 3.8 |
| Lasso's C | Chosen on validation sets | 3.11 |
| Timing | Each selection fit 10 times; a searching method searches once, then is timed 10 times | 3.7, 3.11 |

## 3.1 The basic flow

One experiment is a *cell*. For each split, on the training set only, the
runner:

1. fills missing values with the training median, and standardizes every
   feature with the training mean and standard deviation;
2. chooses `k` on validation sets, for a method with a `k` list
   (section 3.11);
3. fits the selector on every gene, which returns `k` features;
4. fits the classifier on those features, on the whole training set
   (section 3.8).

The test set is then scored once. Nothing about it reaches the choice of
features or of `k`. Scores are summarized over the splits. They judge the
whole procedure, not one gene list: each split may choose a different
list.

## 3.2 The most common mistake: selecting on all the data

It is tempting to select genes once on all patients, then cross-validate
only the classifier. The error is large. On pure noise (100 patients,
5,000 random genes, random labels, 20 genes, 5-fold cross-validation) the
true AUC is 0.5:

| Where the 20 genes were chosen | Test AUC |
|---|---|
| On all patients, before splitting | 0.96 |
| On the training patients only | 0.44 |
| The truth | 0.50 |

Selecting on all patients picks genes that fit the test patients by
chance. Ambroise and McLachlan (2002) showed the same on real microarray
data. On BreastMNIST with shuffled labels, the F-test picking 50 embedding
dimensions gives a median AUC over 20 splits of 0.465 inside each split,
and 0.645 when selected on all 780 images first
([`selection.ipynb`](../featsel_datasets/medmnist_breastmnist/demos/selection.ipynb)).

## 3.3 Random splits instead of k-fold

Each split draws a fresh random training set with the class proportions
kept (stratified Monte Carlo splits). We do not use k-fold
cross-validation:

- In 5-fold, any two training sets share three quarters of their samples.
  The fold scores are correlated, so their spread understates the real
  variation. Bengio and Grandvalet (2004) proved that no unbiased
  estimator of the variance of k-fold cross-validation is valid for every
  data distribution.
- Five numbers are too few to describe a distribution. Random splits can be
  drawn as often as time allows.
- Leave-one-out was rejected. A one-patient test set has no AUC or
  macro-F1, so only one pooled number remains, with no spread. Training
  sets differ by one patient, so stability reads near 1.0 for every
  method. And it costs one model per patient: 2,976 per cell on SCAN-B.

Random splits still overlap, since all come from the same 2,976 patients.
But the dependence is no longer forced by k-fold's disjoint test sets.

**On SCAN-B, one split serves all five tasks.** Each split is stratified on
the five-class PAM50 label, so every subtype-vs-rest task is scored on the
same patients.

**How many splits.** The first plan was 100 splits at training fractions
0.5, 0.7 and 0.9, for a learning curve. That grid would have taken about
four days. Every run now uses 16 splits with 70% of samples in the
training set. Sixteen or more values support quartiles; five do not.

**More splits help less than they seem.** The corrected test of section 7.3
multiplies the variance by 1/J + n_test/n_train, for J splits. The second
term does not shrink with J, so the 95% interval has a fixed lower limit.
Width as a multiple of that limit (variance factor combined with the t
value for J − 1 degrees of freedom):

| Splits J | 5 | 10 | 16 | 20 | 32 | 100 |
|---|---|---|---|---|---|---|
| Width, 70% for training | 1.72 | 1.28 | 1.16 | 1.13 | 1.08 | 1.02 |

From 5 to 16 splits the interval narrows by about a third. Beyond 16 it
can narrow by at most 14% more. So 16 is not only what the laptop allowed: on
any machine, it gives most of what
more splits can give. A difference not established at 16 splits needs
more patients, not more splits.

We report the **mean over splits**, which estimates the score on new
patients. The first and third quartiles show the spread without assuming
symmetry. The spread shows how much a score moves when the same patients
are split again. It is not a confidence interval for new patients.

**Paired comparison.** All methods run on the same splits, which removes
most split-to-split noise. A method's gain over the random control is its
score minus the control's on the same split, at the same `k`. The
differences are not independent, so a standard paired test would
overstate certainty. Chapter 7 uses the corrected resampled t-test of
Nadeau and Bengio (2003), which widens the variance for the overlap
(section 7.3).

## 3.4 The random control

The control picks `k` features at random, with a new draw on every split.
It is the most trivial selection method. It gets the same splits, `k` list
and classifier as every method.

Its test AUC at each `k` is a curve, and that curve is the bar. A method has
value at a given `k` only if it scores above random at that `k`. Random
selection costs almost nothing, so on the cost axis the curve sits at the
cheap end. At the largest `k`, random selection keeps every feature, so the
curve ends at the all-features reference.

We do not compare a method with random at random's own chosen `k`. More
random features almost always score better, so random would choose its
largest `k`. That comparison would ask whether selection beats many
features, which the all-features reference already answers.

Without the control, a comparison cannot tell whether a method finds signal
or only reduces the number of features. Neekhra et al. (2025) found that small random subsets
matched or beat feature selection on 28 of 30 high-dimensional datasets.
Chapter 7 shows this on our data too once `k` is large.

## 3.5 Scores

The tasks are imbalanced (chapter 2). On Normal vs rest (200 against 2,776
patients), always predicting "rest" has accuracy 0.93, so plain accuracy
is not used.

| Score | What it measures | Role |
|---|---|---|
| **ROC AUC** | The chance that a random positive patient is scored above a random negative one. | Main score. It needs no decision threshold. |
| **Macro-F1** | The mean F1 over classes; the small class weighs as much as the large one. | Side score. |
| **Balanced accuracy** | The mean recall over classes. | Side score. |

F1, accuracy and balanced accuracy depend on a decision threshold, and so
imply a choice of costs between false positives and false negatives. A
comparison of selection methods should not make that choice. AUC needs no
threshold.

AUC is computed per split, from one model on that split's test set.
Pooling scores from different models into one AUC would mix scales.

## 3.6 Stability

A method can score well and still pick different genes on every split.
Its gene list is then an accident of the split, not a finding. We measure stability by Kuncheva's (2007)
consistency index, averaged over all pairs of splits: 1 for identical gene
sets, 0 for chance overlap, so the random control scores about 0. 5 splits
give 10 pairs, 16 give 120, 20 give 190.

The index assumes sets of equal size. For methods that choose their own
count, we use the mean size of each pair, an approximation. So we also
compute the measure of Nogueira, Sechidis and Brown (2018). It is built
from how often each feature is selected and needs no equal sizes. With
equal sizes it equals the mean pairwise Kuncheva index.

A random method needs a different seed on each split. With one seed, its
stability would be a meaningless 1.0.

## 3.7 Cost

For every fit we record:

- **Selection time**: the selector's `fit` alone. Each selection is fit 10
  times by a fresh selector, and the median is recorded. Sequential
  selection takes minutes per fit, so it is fit once.
- **Search time**: choosing `k`, or a method's own hyperparameter, on
  validation sets (section 3.11), recorded apart.
- **Peak memory** of the selection.
- **The device**: CPU or GPU.
- **Reproducibility**: whether every repeated fit chose the same genes.

Classifier training time is not counted: it is the same for every method
at the same `k`. The design planned to state each method's time and memory
complexity next to the measurements. This was not done.

## 3.8 The classifier

L2-penalized logistic regression with C = 1. The penalty is (λ/2)‖w‖², and
scikit-learn's C is 1/λ. Features are standardized on the training set
first, so the penalty treats them alike. The default solver, `lbfgs`, does
not penalize the bias term.

C is fixed. Tuning it would refit the classifier for every value of C on
every validation set, on top of the search for `k`. Every
method gets the same classifier, so the ranking stays fair, but absolute
scores may be slightly lower than with tuning. Lasso's own C is separate
and chosen on validation sets (section 3.11).

## 3.9 Reproducibility

There is one global seed, 42. Split `s` uses seed 42 + `s` for every random
step in it: the random selector, the forest, the solvers and the validation
splits. Every selector that uses randomness seeds its own generators in its
`fit`. Re-running a configuration gives the same genes and scores, for any
number of workers. Only the times change.

## 3.10 Preprocessing is fixed

Every run uses **standardization only**, fit on the training set. Varying
preprocessing would double or quadruple an already large grid, for a
question outside the project's four tasks.

**Min-max scaling was rejected without a test.** Like standardization, it
shifts and scales each feature. So it changes nothing for methods that
ignore scale, such as the F-test, correlation and
trees. For methods that depend on scale, such as k-NN and penalized models
(L1 or L2), it is worse: one outlier sets the range and squeezes every
other sample.

**A side experiment, before the current protocol.** 98.1% of genes reject
normality. A quantile transform maps each feature to a normal distribution
through its ranks, but discards the size of a fold change. On
`scanb_small`, with F-test selection, logistic regression and 10 splits,
standardization was compared with a rank-to-normal transform and
scikit-learn's quantile transform. The two transforms tied. On the five
binary tasks:

- Most differences were within 0.01. Her2 and Normal gained up to 0.011.
  LumA lost up to 0.021; at `k` = 100 it lost on all 10 splits.
- With C tuned on validation sets and scored by AUC, differences stayed
  within 0.004, except LumA at `k` = 100 (−0.017).

So standardization was kept. Shrinkage LDA, which assumes normal classes,
and k-NN, which suffers from heavy tails, might still gain. A run over
every classifier was cancelled, so this is unknown.

If preprocessing is ever varied, it must be varied for every method alike.
A method with its own preprocessing mixes the method with its transform.

## 3.11 The feature budget `k`

On SCAN-B, `k` is 10, 50, 250, 500, 1000, 1500 and 2000. A first run
stopped at 1,000, where AUC was still rising for every method with a `k`
list. On BreastMNIST and FGVC-Aircraft (2,048 features), `k` is 10, 50,
250, 500 and 1000; random selection also runs at 2,048. The list passes the
number of training images, because the goal is the best AUC, not a small
set. A two-stage chain uses the same list, cut below the size its first
step keeps.

- **Why a sweep.** One `k` cannot tell a method good at small budgets from
  one that needs many features.
- **Why the same `k` for every method.** It makes the comparison with the
  random control fair.
- **Why 50.** PAM50 is a 50-gene signature.
- **No selection.** A run on all features is kept as a reference. It shows
  whether selection helps the model or only shrinks it
  ([section 8.7](08_results_frontier.md#87-is-selection-needed-at-all)).

The first design gave `k` as a percent of the total sample count, to
compare budgets across datasets and training fractions. With one training
fraction, `k` is given directly.

**Choosing `k` on validation sets.** Choosing `k` by test AUC would use the
test set twice. So each method with a `k` list chooses it on validation
sets:

1. The training set is cut into 5 folds. Each fold in turn is the
   validation set.
2. The selector is fit on the other four folds.
3. For each `k`, logistic regression is fit on the top `k` features and
   scored by AUC on the validation set.
4. The `k` with the best mean validation AUC wins; a tie goes to the
   smaller `k`.
5. That `k`'s classifier, fit on the whole training set, is scored once on
   the test set.

A method that ranks once fits once per fold and cuts every `k` from that
fit: 5 extra fits, against the 10 timed fits of each selection.

**Sequential selection runs only at `k` = 5 and 10, and does not search.**
Each of its steps tries every remaining feature, with 5-fold
cross-validation, so its cost grows at least linearly with `k`
(section 4.2). On SCAN-B one fit at `k` = 10 already takes minutes; `k` =
500 would take hours per split. A larger `k` needs a filter to cut the
features first.

Standardization is fit on the whole training set, so the validation sets
helped set its means and spreads. That uses no labels, and the test set
stays untouched.

**A searching method searches once.** Lasso chooses its C from four values
by mean AUC over 5 validation sets. RFECV chooses its count by 5-fold
cross-validation on the training set. The 10 timed repeats then use what
the search chose: Lasso with that C, RFECV as RFE to that count. The search
time is recorded on its own. The runner checks that the repeats select the
same features as the search.

## 3.12 Data precision

The data are held as 32-bit floats, which halves their memory. Peak
selection memory on SCAN-B fell by 30% to 45%, so more workers fit on one
machine. SCAN-B's values have at most 7 measured digits, so 32 bits lose
nothing measured.

The U test sums its ranks in 64 bits. An earlier run, scored by macro-F1,
showed why. In 32 bits, the tie correction of
near-constant genes went wrong: Mann-Whitney chose different genes in
every SCAN-B cell, and macro-F1 moved by up to 0.55. After the fix, a
one-split check against 64-bit data moved Mann-Whitney's macro-F1 by at
most 0.007. Lasso, RFE and the forest moved by about 0.01 on average,
because their solvers are sensitive to rounding.

## 3.13 Scores we considered and did not use

Two single-number summaries were rejected:

- **A leave-one-out confusion table.** It leaves out one patient at a time
  and counts the decisions, so it trains on almost all patients.
  But it brings back the decision threshold and gives one table with no
  spread.
- **The area under a score-against-`k` curve,** minus the random control's
  area. It depends on how the `k` axis is scaled, and methods that pick
  their own count give a point, not a curve.

We report each method at its chosen `k`, and the whole curve of test AUC
over `k`, with random selection's curve as the bar. One number hides where along `k` a method wins.

## 3.14 Limits of this protocol

- The spread over splits is not a confidence interval for new patients.
- The five SCAN-B tasks share patients, so they are not independent
  replicates.
- The decision threshold is fixed at 0.5 on the predicted probability.
  Macro-F1 and balanced accuracy depend on it; AUC does not.
- Lasso chooses its penalty on validation sets (section 3.11). The other
  selectors' own settings stay fixed.
- The classifier's C is fixed at 1 (section 3.8).
- The runner stores each cell's metrics, not each test patient's score, so
  a new metric needs a new run. Storing the scores was designed but not
  built (chapter 10).
