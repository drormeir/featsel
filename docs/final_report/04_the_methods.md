# 4. The methods

[Back to contents](README.md) · Previous: [3. How to measure](03_how_to_measure.md) · Next: [5. The software](05_the_software.md)

## 4.1 Three families

Feature selection methods fall into three families (Guyon and Elisseeff
2003; Saeys et al. 2007). They differ in how much they use the classifier.

| Family | How it works | Strength | Weakness |
|---|---|---|---|
| **Filter** | Scores each feature from the data alone, usually one feature at a time, with a statistical test. | Fast; scales to tens of thousands of features. | Ignores redundancy: two genes with the same signal both rank high. |
| **Embedded** | Selects features as a side effect of fitting a model, such as a penalized regression or a forest. | Sees features together; one model fit. | The result depends on the model's settings. |
| **Wrapper** | Searches over subsets, training a classifier for each candidate subset. | Sees interactions; uses the real classifier. | Slowest by far; with few samples it can overfit the search itself. |

Kohavi and John (1997) introduced the wrapper idea. The lasso
(Tibshirani 1996) and random forests (Breiman 2001) are the two embedded
methods used here. The family is a label, not shared code: in `featsel`
each method is its own class.

We add two more groups: **Higher Criticism**, a filter that picks its own
number of features, and the **random control**.

**Multiple testing.** A filter that tests every gene runs thousands of
tests. The Benjamini and Hochberg (1995) procedure keeps the expected
share of false discoveries below a chosen level. Bonferroni correction
controls the chance of even one false discovery, and is stricter. The FDR
and FWE selectors apply these two.

## 4.2 The 15 methods

Every method is a class in `featsel.selectors`. "Ours vs scikit-learn"
says whether it reproduces a scikit-learn method exactly (same genes),
which tests check.

| Method | Family | Idea in one sentence | Main setting | Ours vs scikit-learn |
|---|---|---|---|---|
| RandomSelector | control | Pick `k` features at random. | `k` | — |
| VarianceThreshold | filter | Drop features that barely vary. | threshold | same genes |
| ANOVAFSelector | filter | Rank by the F-test (difference of class means). | `k` | same genes as `SelectKBest(f_classif)` |
| PercentileSelector | filter | F-test, keep the top percent. | percentile | same as `SelectPercentile` |
| FDRSelector | filter | F-test, keep what passes Benjamini-Hochberg false discovery control. | alpha | same as `SelectFdr` |
| FWESelector | filter | F-test, keep what passes Bonferroni. | alpha | same as `SelectFwe` |
| MannWhitneySelector | filter | Rank by the Mann-Whitney U test (section 4.3). | `k` | none in scikit-learn; equals SciPy's test |
| CorrelationSelector | filter | Rank by correlation with the label, then drop one of each highly correlated pair. | `k`, thresholds | own method |
| MutualInfoSelector | filter | Rank by mutual information, which catches any dependence, not only linear. | `k` | same scores as `mutual_info_classif`, bit for bit |
| HigherCriticismSelector | filter, own count | Keep the features below the Higher Criticism threshold (section 4.4). | variant, alpha0 | none in scikit-learn |
| LassoSelector | embedded | L1-penalized logistic regression; keep the largest weights; the penalty sets most weights to exactly zero. | C | same as `SelectFromModel` with liblinear |
| TreeImportanceSelector | embedded | Random forest; rank by how much each feature reduces impurity in the trees. | number of trees | same as `SelectFromModel(RandomForest)` |
| RFESelector | wrapper | SVM-RFE: train a linear SVM, remove the features with the smallest weights, repeat (Guyon et al. 2002). | `k`, step, C | same as `RFE(LinearSVC)` with the default solver |
| RFECVSelector | wrapper | RFE, with the number of features chosen by inner cross-validation. | step, C, folds | same as `RFECV` with the default solver |
| SequentialSelector | wrapper | Greedy forward search: add the one feature that most improves cross-validated accuracy, repeat. | `k`, folds | same as `SequentialFeatureSelector` with the default solver |

Each method has a tutorial notebook:
`featsel/selectors/<name>/demos/tutorial.ipynb`.

### A few methods in more detail

**Lasso.** The penalty C·Σ|wᵢ| drives most weights to exactly zero. The
features with non-zero weights are selected. One consequence matters later:
at a fixed C, Lasso keeps only a few hundred genes on SCAN-B, so asking it
for 1,000 gives fewer.

**Random forest importance.** Each tree is grown on a bootstrap sample of
the patients. At each split point a tree tries a random subset of
√p features and picks the best. A feature's importance is the total drop
in impurity it causes, averaged over the trees.

**SVM-RFE.** A linear SVM's weight on a feature says how much the decision
depends on it, on standardized data. RFE removes the weakest features,
retrains, and repeats. The `step` sets how many go per round: 0.1 removes
10% of the remaining features each round. Guyon et al. (2002) introduced
it on gene expression data from leukemia and colon cancer, and it is a
widely used wrapper for such data.

**Sequential forward selection.** It starts empty. At each step it tries
every remaining feature, scores each by cross-validation, and adds the
best. For 10 features out of 28,552 that is about 285,000 candidates, each
scored by 5-fold cross-validation: about 1.4 million model fits. It is the
most expensive method here.

### One ranking for every `k`

Mann-Whitney, Lasso and tree importance rank all features in one fit, and
that fit does not depend on `k`. So their top 10, top 50 and top 250 can
be cut from one ranking per split. The cut equals a separate fit at each
`k`.

RFE needs one more step. It is fit once, down to the smallest `k`. With a
fixed step, the rounds for a larger `k` are the same up to that `k`'s last
round. So ranking the features by when they were removed, and within a
round by their weight, gives RFE's selection at every larger `k`. Tests
check that every cut equals a separate fit. Chapter 6 shows the time this
saves.

## 4.3 Why we score genes with the U test, not the F-test

This was one of the main decisions of the project (2026-10-25).

**The data.** Chapter 2 showed two facts: only the order of a gene's values
is meaningful, and about half the values sit tied at a floor. Within a
class, a gene is a pile of identical floor values plus a spread of detected
values. It is not a normal curve.

**The F-test** compares class means. Its result depends on the size of the
values, so it would change under another, equally valid transform. Its
p-values assume normal values with equal spread in both classes. The F
statistic still ranks genes sensibly, but its tiny p-values, which Higher
Criticism reads, rest on an assumption the data breaks.

**The Mann-Whitney U test** (Mann and Whitney 1947) uses only the order of
the values, and assumes no normal curve. For a binary task, U divided by
n₁n₂ is exactly the AUC of the gene used alone as a score (Hanley and
McNeil 1982). So ranking genes by U-test p-value ranks them by how well
each one alone separates the classes. Its p-value answers the question we
want to ask: how likely is a separation this good if the labels were
shuffled, with the class sizes fixed? Ties, such as the floor, get average
ranks, and the floor is treated as what it is: one large group of tied
values. With two classes, the U test is the same test as Kruskal-Wallis,
"ANOVA on ranks". It needs exactly two classes, which matches the
project's binary tasks.

The p-value is the tie-corrected normal approximation, as in SciPy.

So every filter in the experiments uses the U test. One test for all
filters makes the comparison a comparison of selection rules, not of tests.
The F-test selectors stay in the package as drop-in replacements for
scikit-learn's, but are not used in the experiments.

**Its cost.** Before the speed work of chapter 6, a U-test fit took about
10 s on SCAN-B, against about 1 s for the F-test. Mann-Whitney ranks once
per split, so the cost is paid once per split, not once per `k`.

**Its limit.** The normal approximation is not exact in the extreme tail,
which is where HC2004 reads its p-values. With ties, the exact
permutation p-value depends on each gene's own pattern of ties. So it
cannot be computed once for all genes.

**Numerical care.** Many genes have p-values below 10⁻³⁰⁸, where a
standard 64-bit float rounds to 0. Higher Criticism needs to tell them
apart. We compute the logarithm of the p-value directly from the normal
tail, so nothing underflows ([`featsel/selectors/log_pvalue/`](../../featsel/selectors/log_pvalue/)).

**A better model, not built.** Each gene could be described by four
numbers: the two class sizes, and the share of each class at the floor.
That gives an exact null distribution per gene, separating "how often the
gene is detected" from "how high it is when detected". It needs a separate
computation per floor count and is left for future work
([Appendix D.4](appendix_d_data_details.md#d4-a-better-model-for-floor-heavy-genes-not-built)).

## 4.4 Higher Criticism

Most filters need `k` from the user. Higher Criticism (HC) picks its own
count. It comes from detecting rare, weak signals among many tests (Donoho
and Jin 2004, 2008).

**The idea.** Sort the N p-values: p₍₁₎ ≤ p₍₂₎ ≤ … ≤ p₍N₎. If no gene
carried signal, the p-values would be uniform, and p₍ᵢ₎ would sit near
i/(N+1). HC measures, at each i, how far the sorted p-values fall below
that line, in units of their standard deviation. Where this gap is largest,
the signal stands out most. HC keeps every feature up to that point.

We implement two forms. The first is exactly as published. The second
changes one detail of the published form:

| Variant | Statistic | Reference |
|---|---|---|
| HC2004 | HC(i) = √N · (i/N − p₍ᵢ₎) / √(p₍ᵢ₎(1 − p₍ᵢ₎)) | Donoho and Jin (2004), section 1.2 |
| HC2008 | HC(i) = √N · (μᵢ − p₍ᵢ₎) / √(μᵢ(1 − μᵢ)), μᵢ = i/(N+1) | HC thresholding, Donoho and Jin (2008, 2009). The papers use μᵢ = i/N, an approximation (2009, Definition 1.2). We use i/(N+1), the exact mean of the i-th sorted uniform p-value. |

The change from i/N to i/(N+1) is small for the large N here. It is not
in the papers, and we have not measured how much it moves the selected
count.

The maximum is searched over the smallest alpha0 · N p-values. Both papers
use alpha0 = 0.1, and so do we.

**When HC is expected to work.** Its guarantees hold when the useful
features are *rare* (a small share of all features), *weak*, and
*independent*. Donoho and Jin (2004) built HC to detect this "rare and
weak" setting. Donoho and Jin (2008, 2009) turned it into a selection
threshold and proved it optimal there. Both papers derive how the sorted
p-values behave under the null from independent p-values. Correlated
genes give correlated p-values. Chapter 2 showed that neither of our datasets meets these
conditions. Chapter 7 shows what HC does then.

**Decision: alpha0 stays 0.1** (2026-10-31). On SCAN-B the HC score is
still rising at the edge of its search. Raising alpha0 because we know the
signal is widespread would tune the method to the answer. A selector must
not assume what it is meant to find.

## 4.5 What the experiments used

| Experiment | Methods |
|---|---|
| The comparison (chapter 7) | Random, Mann-Whitney, Lasso, tree importance, RFE, HC2004, HC2008 |
| The frontier (chapter 8) | Random, Mann-Whitney, HC2008, correlation (Spearman), mutual information, Lasso (two solvers), tree importance, RFE (two solvers), RFECV, Sequential |
| The classifier study (chapter 7) | Random and Mann-Whitney, with five classifiers |

Variance threshold, ANOVA, Percentile, FDR and FWE are in the package and
tested, but not in the experiments: they use the F-test, which the U test
replaces (section 4.3).
