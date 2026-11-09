# 3. How to measure a selection method

[Back to contents](README.md) · Previous: [2. The data](02_the_data.md) · Next: [4. The methods](04_the_methods.md)

A selection method cannot be judged by looking at the genes it picks. We
judge it by what those genes do for a classifier on patients it has not
seen. This chapter explains how to do that without fooling ourselves.

## 3.1 The basic flow

For one experiment, called a *cell*:

1. Split the patients into a training part and a test part.
2. Fit the selection method on the training part only. It sees every gene,
   but only the training patients. It returns `k` genes.
3. Train a classifier on the training patients, using only those genes.
4. Score the classifier on the test patients.

Repeat over many random splits, and summarize the scores.

The score judges the whole procedure (select, then train), not one gene
list. Each split may choose a different list.

## 3.2 The most common mistake: selecting on all the data

It is tempting to select genes once on all patients, and then
cross-validate the classifier. This is wrong, and the error is large.

We showed it on pure noise: 100 patients, 5,000 random genes, random
labels, 20 genes selected. The true AUC is 0.5, since there is nothing to
find.

| Where the 20 genes were chosen | Test AUC |
|---|---|
| On all patients, before splitting | 0.96 |
| On the training patients only | 0.44 |
| The truth | 0.50 |

Selecting on all patients picks genes that happen to fit the test patients
by chance, and the test then rewards that chance. Ambroise and McLachlan
(2002) showed the same on real microarray data. In `featsel` the selector
is always fit inside the split. Imputation and scaling are fit inside the
split too.

Source: [`docs/selection_strength.md`](../selection_strength.md).

## 3.3 Random splits instead of k-fold

We use stratified Monte Carlo splits: each split draws a fresh random
training set, with the class proportions kept. We do not use k-fold
cross-validation.

- In 5-fold, any two training sets share three quarters of their samples.
  The fold scores are strongly correlated, so their spread understates the
  real variation. Bengio and Grandvalet (2004) proved that no unbiased
  estimator of the variance of k-fold cross-validation exists.
- Five numbers are too few to describe a distribution. Random splits can be
  drawn as many times as time allows.
- Leave-one-out was rejected: each test set is one patient, so no AUC can
  be computed per split, and it costs one model per patient.

We report the **median over splits with the first and third quartiles**,
or the mean with the standard deviation where noted.

**What the spread means.** The spread measures how much a score moves when
the same patients are split again. It is not a confidence interval for new
patients, because every split reuses the same cohort.

**Paired comparison.** All methods run on the same splits. So two methods
can be compared split by split, which removes most of the split-to-split
noise. The gain of a method over the random control is its score minus the
random control's score on the same split and at the same `k`.

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
| **ROC AUC** | The chance that the model scores a random positive patient above a random negative one. | It needs no decision threshold, so it does not mix the method's quality with a choice of costs. Main score of the frontier (chapter 8). |
| **Macro-F1** | The average of the F1 score of each class. | It weights the small class as much as the large one. Main score of the comparison (chapter 7). |
| **Balanced accuracy** | The average of the recall of each class. | A simple imbalance-robust check. |

Accuracy, MCC and G-mean are also recorded in the comparison.

AUC is computed per split, from one model on that split's test patients.
Pooling scores from different models into one AUC would compare numbers
that are not on the same scale.

## 3.6 Stability

A method can score well and still pick different genes on every split.
Its gene list is then an accident of the split, not a finding.

We measure stability by Kuncheva's (2007) consistency index between the
gene sets of two splits, averaged over all pairs of splits. It is 1 for
identical sets and 0 for the overlap expected by chance. So the random
control scores about 0.

## 3.7 Cost

For every fit we record:

- **Selection time**: the time of the selector's `fit` alone, not of the
  classifier. In the frontier each selection is fit 10 times by a fresh
  selector, and the median is recorded (once for Sequential selection,
  which takes minutes).
- **Peak memory** of the selection.
- **The device**: CPU or GPU.
- **Reproducibility**: whether every repeated fit chose the same genes.

## 3.8 Tuning the classifier without leakage

The classifier, L2-penalized logistic regression, has a strength parameter
C. In the comparison (chapter 7) C is tuned inside each split: the
training part is split again three times, each C from {0.01, 0.1, 1, 10,
100} is scored by AUC on the inner validation parts, and the best C is
refit on the whole training part. The test part is used once, at the end.

The selector is fit once on the whole training part, not inside each inner
split. The inner validation patients therefore helped choose the genes,
which can bias the choice of C slightly. The test part stays untouched, so
the reported scores stay honest.

In the frontier (chapter 8) C is fixed at 1, to keep the run short. Every
method gets the same classifier, so the ranking stays fair, but the
absolute scores may be a little lower than with tuning.

## 3.9 Reproducibility

One global seed, 42. Split `s` uses seed 42 + `s` for every random step in
it. Every selector seeds its own random generators in its `fit`. Re-running
a configuration gives the same genes and the same scores; only the times
change.

## 3.10 Limits of this protocol

- The spread over splits is not a confidence interval for new patients.
- The five SCAN-B tasks share patients, so they are not independent
  replicates.
- The decision threshold is not tuned: 0.5 on the predicted probability.
  Macro-F1 and balanced accuracy depend on it; AUC does not.
- The selectors' own settings, such as Lasso's penalty, are fixed.
- Choosing a "best `k`" from test scores would leak. We report whole curves
  over `k` instead.
