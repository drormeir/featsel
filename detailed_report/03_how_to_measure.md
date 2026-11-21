# 3. How to measure a selection method

[Back to contents](README.md) · Previous: [2. The data](02_the_data.md) · Next: [4. The methods](04_the_methods.md)

A selection method is judged by what its genes do for a classifier on
patients it has not seen. This chapter describes the protocol of every
method comparison. A few side studies keep their own settings, stated
where they appear.

| Part | Choice | Section |
|---|---|---|
| Splits | 16 random stratified splits, 70% of samples in the training set | 3.3 |
| Main score | ROC AUC: mean over splits, quartiles for spread | 3.3, 3.5 |
| Side scores | Macro-F1, balanced accuracy | 3.5 |
| Baseline | Random features at each `k`: a method has value at `k` only above them | 3.4 |
| `k` | Chosen on validation sets: 5 folds of the training set, best mean AUC, tie to the smaller `k` | 3.11 |
| Classifier | Logistic regression, C fixed at 1 | 3.8 |
| Lasso's C (its penalty, section 4.2) | Chosen on validation sets | 3.11 |
| Timing | Each selection fit 10 times (3 on FGVC-Aircraft, once in the simulation of section 8.11, which reports no times; Sequential once on SCAN-B and BreastMNIST); a searching method searches once, then is timed | 3.7, 3.11 |

## 3.1 The basic flow

One experiment is a *cell*: one task, split, method, `k` and classifier.
On each split, on the training set only, the runner:

1. fills missing values with the training median, and standardizes every
   feature with the training mean and standard deviation;
2. chooses `k` on validation sets, for a method with a `k` list
   (section 3.11);
3. fits the selector on every gene, which returns `k` features;
4. fits the classifier on those features, on the whole training set
   (section 3.8).

Nothing about the test set reaches the choice of features or of `k`. The
runner scores it at every `k` in the list, to draw the full curve
(section 3.13); a method's result is the score at its chosen `k`. Scores are summarized over the splits. They judge the
whole procedure, not one gene list: each split may choose a different
list.

## 3.2 The most common mistake: selecting on all the data

It is tempting to select genes once on all patients, then cross-validate
only the classifier. The error is large. Ambroise and McLachlan (2002)
showed it on real microarray data. On BreastMNIST with shuffled labels,
where nothing can be learned and the true AUC is 0.5, the F-test (section
4.3) picking
50 embedding dimensions gives a median AUC over 20 splits of 0.465 inside
each split, and 0.645 when selected on all 780 images first
([`selection.ipynb`](../featsel_datasets/medmnist_breastmnist/demos/selection.ipynb)).

Selecting on all the samples picks features that fit the test samples by
chance.

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
  macro-F1 (section 3.5), so only one pooled number remains, with no
  spread. Training sets differ by one patient, so stability (section 3.6)
  reads near 1.0 for every
  method. And it costs one model per patient: 2,976 models per method and
  `k` on SCAN-B.

Random splits still overlap, since all come from the same 2,976 patients.
The corrected test below allows for this overlap.

**On SCAN-B, one split serves all five tasks.** Each split is stratified on
the five-class PAM50 label, so every subtype-vs-rest task is scored on the
same patients.

**How many splits.** The first plan was 100 splits at training fractions
0.5, 0.7 and 0.9, for a learning curve. It was dropped for time. Every
method comparison now uses 16 splits with 70% of
samples in the training set: 50% trains on fewer samples than a model
in real use would have, and 90% leaves too few to test: on BreastMNIST,
59 images, 18 of them malignant. Sixteen or more values support quartiles; five do not.

**More splits help less than they seem.** The corrected test below
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
more splits can give. A difference not established (p < 0.05 under the
corrected test below) at 16 splits needs
more patients, not more splits.

We report the **mean over splits**, which estimates the score on new
patients. The first and third quartiles show the spread without assuming
symmetry. The spread shows how much a score moves when the same patients
are split again. It is not a confidence interval for new patients.

**Paired comparison.** All methods run on the same splits, which removes
most split-to-split noise. A method's gain over the random control is its
score minus the control's on the same split, at the same `k`. The
differences are not independent, so a standard paired test would
overstate certainty. Chapters 7 and 8 use the corrected resampled t-test
of Nadeau and Bengio (2003), which widens the variance for the overlap.

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
Sections 7.2 and 8.2 show this on our data too once `k` is large.

## 3.5 Scores

Four of the five tasks are imbalanced (chapter 2). On Normal vs rest (200 against 2,776
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
threshold, so nothing is tuned for it. The side scores use the
classifier's default threshold, a probability of 0.5, not tuned. On
imbalanced tasks this threshold favors the large class, so the side
scores are low in absolute terms. They serve only to check that the order
of methods agrees with AUC (sections 7.12 and 8.10).

AUC is computed per split, from one model on that split's test set.
Pooling scores from different models into one AUC would mix scales.

## 3.6 Stability

A method can score well and still pick different genes on every split.
Its gene list is then an accident of the split, not a finding. We measure
stability by the measure of Nogueira, Sechidis and Brown (2018) over all
splits: 1 for identical gene sets, 0 for chance overlap, so the random
control scores about 0. It is built from how often each feature is
selected, so it needs no equal sizes. That matters for the methods that
choose their own count.

`featsel.analysis` also has Kuncheva's (2007) consistency index, averaged
over all pairs of splits. It assumes sets of equal size, so the results
do not use it. With equal sizes the two measures agree.

A random method needs a different seed on each split. With one seed, its
stability would be a meaningless 1.0.

## 3.7 Cost

For every fit we record:

- **Selection time**: the selector's `fit` alone. Each selection is fit 10
  times by a fresh selector (3 times on FGVC-Aircraft, a count set by hand; once in the
  simulation of section 8.11), and the median is
  recorded. Sequential selection takes minutes per fit on SCAN-B, and 10
  fits would take about a week (the SCAN-B run with one fit took 18 hours,
  section 7.11), so there and on BreastMNIST it is fit once;
  on FGVC-Aircraft it is fit 3 times, like every method there. On
  FGVC-Aircraft every fit chose the same features, so the count changes
  only how precise the time is, not the AUC.
- **Search time**: choosing `k`, or a method's own hyperparameter, on
  validation sets (section 3.11), recorded apart.
- **Peak memory** of the selection: what Python and NumPy allocate on
  the CPU during the first timed fit, traced by `tracemalloc`. It does not
  count GPU memory or memory that compiled libraries allocate outside
  NumPy (section 7.12).
- **The device**: CPU or GPU.
- **Reproducibility**: whether every repeated fit chose the same genes.

Classifier training time is not counted: it is the same for every method
at the same `k`. Section 4.7 gives each method's cost in time and memory,
as it follows from the algorithm.

## 3.8 The classifier

L2-penalized logistic regression with C = 1, scikit-learn's default. A side study with four other
classifiers is the evidence for it (section 7.7). The penalty is (λ/2)‖w‖², and
scikit-learn's C is 1/λ. Features are standardized on the training set
first, so the penalty treats them alike. The default solver, `lbfgs`, does
not penalize the bias term.

C is fixed. Tuning it would refit the classifier for every value of C on
every validation set, on top of the search for `k`. Every
method gets the same classifier, so the ranking stays fair, but absolute
scores may be slightly lower than with tuning. Lasso's own C is separate
and chosen on validation sets (section 3.11).

**What a different C would change, so far.** A side study scored the same
selections with C = 0.01, 0.1, 10 and 100, on the main splits, for
Mann-Whitney and all features, each C choosing its own `k`
(`configs/clf_c_*.yaml`). On BreastMNIST (still with its copied images,
section 10.1) and FGVC-Aircraft no difference
from C = 1 is established: the largest gain is 0.005 AUC (p = 0.29), and
C = 100 loses 0.011 to 0.014 on BreastMNIST (p = 0.055 and 0.057). SCAN-B
was not tested. Tables: `clf_c_breastmnist70.csv` and
`clf_c_aircraft70.csv` in [`figures/`](figures/).

## 3.9 Reproducibility

There is one global seed, 42. Split `s` uses seed 42 + `s` for every random
step in it: the random selector, the forest of tree importance (section
4.2), the solvers and the
validation folds for `k`. Lasso's search for C and RFECV's search for its
count use their own 5 folds, not shuffled, so they need no seed. Every selector that uses randomness seeds its own generators in its
`fit`. Re-running a configuration gives the same genes and scores, for any
number of workers. Only the times change.

## 3.10 Preprocessing is fixed

Every run uses **standardization only**, fit on the training set. Varying
preprocessing would double or quadruple an already large grid, for a
question outside the project's four tasks.

**Min-max scaling was rejected without a test.** Like standardization, it
shifts and scales each feature. So it changes nothing for methods that
ignore scale, such as the F-test, correlation and
trees. For methods that depend on scale, such as k-nearest neighbours
(k-NN) and penalized models
(L1 or L2), it is worse: one outlier sets the range and squeezes every
other sample.

**A quantile transform was not tested.** 98.1% of genes reject
normality. A quantile transform maps each feature to a normal
distribution through its ranks, but discards the size of a fold change.
Shrinkage linear discriminant analysis (LDA), which assumes normal classes, and k-NN, which suffers from
heavy tails, might gain from it (both are in the classifier study of
section 7.7). This is unknown.

If preprocessing is ever varied, it must be varied for every method alike.
A method with its own preprocessing mixes the method with its transform.

## 3.11 The feature budget `k`

On SCAN-B, `k` is 10, 50, 250, 500, 1000, 1500 and 2000; random selection
also runs at 28,541, every gene. A first run stopped at 1,000, where AUC
was still rising for every method with a `k` list. The list stops at
2,000 because there the best method already reaches or passes the AUC of
all genes on every task (Appendix C.1). On LumA and Normal, AUC still
rises a little at 2,000 (mutual information, section 4.2: 0.958 to 0.960 and 0.945 to
0.948 from 1,500), so a larger `k` could still add a little there. On
BreastMNIST and
FGVC-Aircraft (2,048 features), `k` is 10, 50, 250, 500 and 1000; random
selection also runs at 2,048. The largest `k`, 1,000, is more than the
number of training images (409 and 840), because the goal is the best
AUC, not a small set. A two-stage chain (a filter, then a slower method;
section 5.4) uses the same list, cut below the size its first step keeps:
1,000 genes on SCAN-B and 500 features on the image data, set by hand
(sections 7.9 and 8.7).

- **Why a sweep.** One `k` cannot tell a method good at small budgets from
  one that needs many features.
- **Why the same `k` for every method.** It makes the comparison with the
  random control fair.
- **Why 50.** PAM50 is a 50-gene signature.
- **No selection.** A run on all features is kept as a reference. It shows
  whether selection helps the model or only shrinks it
  (sections [7.3](07_results_scanb.md#73-selection-against-all-genes) and [8.3](08_results_images.md#83-selection-against-all-features)).

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
5. That `k`'s classifier, fit on the whole training set, gives the
   method's test score. (The test set is scored at every `k`, for the
   curves, but only the chosen `k`'s score is the method's result.)

A method that ranks once fits once per fold and cuts every `k` from that
fit: 5 extra fits, against the 10 timed fits of each selection (3 on FGVC-Aircraft).

**Sequential selection runs only at `k` = 5 and 10, and does not search.**
Its cost grows at least linearly with `k` (section 4.2).

Standardization is fit on the whole training set, so the validation sets
helped set its means and spreads. That uses no labels, and the test set
stays untouched.

**A searching method searches once.** Lasso chooses its C from four values
by mean AUC over 5 validation sets. RFECV chooses its count by 5-fold
cross-validation on the training set. The timed repeats then use what
the search chose: Lasso with that C, RFECV as RFE to that count. The search
time is recorded on its own. The runner checks that the repeats select the
same features as the search.

## 3.12 Data precision

The data are held as 32-bit floats, which halves their memory
(section 5.5 gives the effect on a run). SCAN-B's values have at most 7 measured digits, so 32 bits lose
nothing measured.

The U test (section 4.3) sums its ranks in 64 bits. An earlier run
showed why: in 32 bits, the tie correction of near-constant genes went
wrong, and Mann-Whitney chose different genes in every SCAN-B cell. That
run and the check after the fix were not kept, so their scores are not
given here.

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

The limits of this protocol are listed with the others in section 9.6.
