# Discussion

Draft. It interprets the first complete run in chapter 6. The numbers are in
that chapter's generated tables. This chapter does not repeat them.

## Selection matters most when the budget is small

On both datasets the gain over random selection is largest at k = 10. It
shrinks as `k` grows. By k = 250 it is a few hundredths of macro-F1.

The reason is that the signal is spread over many features. On SCAN-B the
subtype labels are computed from 50 genes. But each of those genes moves
with hundreds of others (chapter 3). A random draw of a few hundred genes
therefore already contains good proxies for most of them. On BreastMNIST
<!-- fact: bm_bh_share_min -->24%<!-- /fact --> to <!-- fact: bm_bh_share_max -->28%<!-- /fact --> of the embedding dimensions pass the U test at a 5%
false discovery rate (chapter 3). Again, random draws find signal easily.

This agrees with Neekhra et al. (2025): small random subsets often match
selected ones on high-dimensional data. It also shows why the random control
is needed. Without it, every method at k = 250 would look good. Against it,
the methods differ mainly at small `k`.

## Which methods win where

**Mann-Whitney, Lasso and tree importance form the top group** on both
datasets. Their ranges overlap at most `k`. Five splits cannot rank them.
Tree importance is the only one of the three with a range wholly above
another's: above Lasso at k = 250 on SCAN-B.

**The simplest method holds its own.** Mann-Whitney scores each feature
alone and ignores redundancy. It still sits in the top group. It is also
the most stable method on both datasets: it picks nearly the same features
on every split. This parallels Dudoit et al. (2002), where simple
classifiers did as well as complex ones on tumour gene expression.

**Similar scores, different features.** Lasso, tree importance and RFE
choose quite different features on each split, yet score close to
Mann-Whitney. Many features are interchangeable. So a selected list should
not be read as *the* genes of a subtype. Stability has to be reported next
to accuracy, as Kuncheva's index is here.

**RFE is the weakest on SCAN-B.** At small `k` it fails on LumA and Normal
and is no better than random at k = 250. On BreastMNIST it is weakest at
k = 10 but catches up. One likely cause is the step of 0.5: a single SVM fit
on all <!-- fact: scanb_loaded_genes -->28,552<!-- /fact --> genes removes half of them at once. That fit uses C = 0.01,
fixed. On BreastMNIST there are far fewer features, so each removal matters
less. But an earlier run with step 0.1 gave about the same scores
(chapter 6), so the step is not the whole cause.

**Lasso keeps fewer than `k` features at k = 250** on some tasks. Its
penalty sets most weights to zero. Its score at that `k` is therefore the
score of a smaller set.

## Higher Criticism

Higher Criticism did not behave as a method that picks its own count.
HC2008 stopped at the edge of its search on every task and split of both
datasets. At that size it equals Mann-Whitney, because both use the same
U-test ranking. Its gain over random selection of as many features is small.
HC2004 kept one to a few features and scored far below the rest.

This is not a fault of the implementation. HC's guarantees hold when the
useful features are rare and weak, and independent (Donoho and Jin 2008,
2009). Neither dataset meets either condition (chapter 3). The signal is
widespread, and genes and embedding dimensions are correlated. In that
setting the HC score keeps rising up to the search limit. So the limit, not
the data, sets the count.

The project kept alpha0 at 0.1, the value of both papers. Raising it because
the signal is known to be widespread would tune the method to the answer.
The finding is itself a result: on these two datasets, HC thresholding
reduces to a top-k filter. A dataset with rare, weak signal is where HC
would be expected to differ from the fixed-`k` methods.

## Prediction without the defining genes

The PAM50 labels are computed from 50 known genes. So on SCAN-B we can check
whether a method finds them. Every method finds them much faster than
chance, and Mann-Whitney finds them fastest (`docs/pam50_recovery.md`). But
at k = 50 every method recovers only a small share of the 50, while scoring
well. Good prediction does not require the genes that define the label.
Correlated genes carry the same information. This is the same point as
above: a selected list is one of many that predict equally well.

## The second dataset

The pattern on BreastMNIST is the same as on SCAN-B. Every selector beats
random at every `k` in the median, the gain is largest at k = 10, and HC2008
again stops at its edge. The ranking within the top group differs: Lasso
leads at k = 250 on BreastMNIST. With one task and five splits, the ranges
there are wide. The main conclusions carry over from gene expression to
image embeddings. The fine ranking does not.

## Speed

The runner is faster than plain scikit-learn mainly because it does less
work. It prepares each split once and fits each ranking selector once for
all `k`. Scikit-learn refits imputation, scaling and the selector for every
`k`. The test scores are identical, so the saving costs nothing in
accuracy.

Parallel work helped little in the benchmark. The grid was small, and
starting worker processes took most of the parallel run's time. On the full
datasets each fit is longer, so start-up weighs less. That case was not
measured.

## Threats to validity

- **Few splits.** Five splits give a rough spread. The range is not a
  confidence interval. It measures re-splitting the same patients, not new
  patients.
- **Shared patients.** The five SCAN-B tasks use the same patients and the
  same splits. They are not independent replicates.
- **Computed labels.** PAM50 subtypes are computed from gene expression, not
  measured. The SCAN-B tasks reproduce a published classifier. Selection
  scores on them may be higher than on a measured clinical outcome.
- **One classifier.** All scores use L2-penalized logistic regression. A
  method's rank may change with another classifier.
- **Fixed selector settings.** Lasso, RFE and the forest use fixed settings.
  Only the classifier's C is tuned. RFE's coarse step is chosen for speed.
- **Mild tuning bias.** The selector is fit on the whole training half, so
  the inner validation parts that choose C helped choose the features. The
  outer test half is untouched.
- **Timings.** On SCAN-B the rows were computed in several passes with
  different numbers of workers. Selection times there are not comparable
  across methods. The speed benchmark used a subset of the genes.

## Limitations

The study compares methods at three values of `k` and five splits. More
splits would narrow the ranges. More values of `k` would show where each
method's gain over random selection ends. Both are planned as the next step
of depth. The full grid on SCAN-B, with every selector and classifier, was
not run: it takes about four days.
