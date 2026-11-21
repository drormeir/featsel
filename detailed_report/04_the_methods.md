# 4. The methods

[Back to contents](README.md) · Previous: [3. How to measure](03_how_to_measure.md) · Next: [5. The software](05_the_software.md)

## 4.1 Three families

Feature selection methods fall into three families (Guyon and Elisseeff
2003; Saeys et al. 2007). They differ in how much they use the classifier.

| Family | How it works | Strength | Weakness |
|---|---|---|---|
| **Filter** | Scores each feature from the data alone, usually one at a time, with a statistical test. | Fast; scales to tens of thousands of features. | Ignores redundancy: two genes with the same signal both rank high. |
| **Embedded** | Selects features while fitting a model, such as a penalized regression or a forest. | Sees features together; one model fit. | The result depends on the model's settings. |
| **Wrapper** | Trains a classifier on each candidate subset. | Sees interactions; uses the real classifier. | Slowest by far. With few samples it can overfit the search itself. |

Kohavi and John (1997) is the standard reference for wrappers. The lasso
(Tibshirani 1996) and random forests (Breiman 2001) are the two embedded
methods used here. The family is a label, not shared code: in `featsel`
each method is its own class.

We add two groups. The first is **Higher Criticism** and its relative
**Berk-Jones**: filters that pick their own number of features. The second
is the **random control**.

**Multiple testing.** A filter that tests every gene runs thousands of
tests. The Benjamini and Hochberg (1995) procedure keeps the expected share
of false discoveries below a chosen level. Bonferroni correction controls
the chance of even one false discovery, the family-wise error, so it is
stricter. The FDR (false discovery rate) and FWE (family-wise error)
selectors apply these two corrections.

## 4.2 The 16 methods

Every method is a class in `featsel.selectors`. The last column says
whether tests show it selects the same genes as a scikit-learn method.
Where scikit-learn's solver stops early, the reference is that solver run
to a tight tolerance.

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
| BerkJonesSelector | filter, own count | Keep the features below the exact Berk-Jones threshold (section 4.5). | alpha0 | none in scikit-learn |
| LassoSelector | embedded | L1-penalized logistic regression; the penalty sets most weights to exactly zero; keep the features with a non-zero weight. | C, or values of C to choose from by validation AUC | same as `SelectFromModel` with liblinear (scikit-learn's default solver for L1 and linear-SVM models; section 6.2) run to a tight tolerance |
| TreeImportanceSelector | embedded | Random forest; rank by how much each feature reduces impurity in the trees. | number of trees | same as `SelectFromModel(RandomForest)` |
| RFESelector | wrapper | SVM-RFE: train a linear SVM, remove the features with the smallest weights, repeat (Guyon et al. 2002). | `k`, step, C | same as `RFE(LinearSVC)` with a converged SVM |
| RFECVSelector | wrapper | RFE, with the number of features chosen by cross-validation on the training set. | step, C, folds | same as `RFECV` with a converged SVM |
| SequentialSelector | wrapper | Greedy forward search: add the feature that most improves cross-validated accuracy, repeat. | `k`, folds | same as `SequentialFeatureSelector` with the default solver |

Each method has a tutorial notebook:
`featsel/selectors/<name>/demos/tutorial.ipynb`.

**Settings in the experiments.** The configs fix these values for every
dataset. They were set by hand and not tuned:

- RFE and RFECV: the SVM's C = 0.01 (section 7.9 tests 0.001 to 1), a
  step of 0.1.
- Tree importance: 200 trees (section 7.4: its ranking still depends on
  the forest's seed at this count).
- Correlation: Spearman (the selector's default is Pearson; section 4.6
  gives the reason), and the
  default thresholds. A feature must have a target correlation of at
  least 0.1, and of a pair above 0.95, one is dropped.

Only Lasso's C is chosen, from 0.01, 0.1, 1 and 10, a list also set by
hand (section 3.11).

### A few methods in more detail

**Lasso.** The penalty λ·Σ|wᵢ| drives most weights to exactly zero. The
features with non-zero weights are selected. scikit-learn sets the
penalty's strength by C = 1/λ, so a smaller C keeps fewer features.

So C, not the requested count, caps the count. An earlier SCAN-B
comparison, with C fixed at 1 and liblinear as the solver, showed this.
Asked for 1,000 genes, Lasso kept 127 to 647 on four of the five tasks,
and 883 to 1,000 on LumA.

liblinear at its default tolerance stops before the solution, and keeps
more genes than the exact solution (section 6.2). LassoSelector now
solves exactly, with PPDNA, a dual Newton proximal point algorithm (Liu
and Zhou 2023; section 6.2), and handles classification
only. In the experiments it takes no `k`. It chooses C by validation AUC
(section 3.11), and that C sets the count.

**Random forest importance.** Each tree is grown on a bootstrap sample of
the patients. At each split point a tree tries a random subset of
√p features and picks the best. A feature's importance is the total drop
in impurity it causes, averaged over the trees.

**SVM-RFE.** On standardized data, a linear SVM's weight on a feature says
how much the decision depends on it. RFE removes the weakest features,
retrains, and repeats. It uses an exact Newton solver only (section 6.2). The `step`
sets how many features go per round. A step of 0.1 removes 10% of the
starting features each round, so about ten rounds. Guyon et al. (2002)
introduced it on gene expression data from leukemia and colon cancer. It
is a widely used wrapper for such data.

**Sequential forward selection.** It starts empty. At each step it scores
every remaining feature by cross-validation and adds the best. For 10
features out of 28,541 that is about 285,000 candidates, each scored by
5-fold cross-validation: about 1.4 million model fits, the most of any
method here. The count grows with `k`, so we run it only at `k` = 5 and 10
(section 3.11). It scores candidates by accuracy: scikit-learn's default,
and the only score the batched Newton solver (section 6.2) computes.
Section 3.5 rejects accuracy as a judge of methods; here it only ranks
candidates, and AUC still judges the result. On imbalanced tasks, mostly
Normal, accuracy may favor features that help the large class, so
Sequential may be underrated.

**The wrappers use different classifiers inside.** RFE and RFECV rank by a
linear SVM (`LinearSVC`). Sequential scores candidates with logistic
regression. Every method is then judged by logistic regression. Part of a
gap between RFE and Sequential may come from this difference.

**Correlation: a defect, fixed after the runs.** Of each highly
correlated pair, CorrelationSelector dropped the gene less correlated with
the label, even when the other gene of the pair was itself dropped. Take
genes A, B and C, in falling order of correlation with the label. A–B and
B–C correlate above the threshold, A–C does not. Then B was dropped for
A, and C for B. Only A remained, though C was redundant only with a gene
that was gone. The selector now walks the genes from the largest
correlation with the label down, and drops a gene only when it is
redundant with a gene already kept, so A and C stay. A test checks this
example. Every correlation result in this report comes from the old
version: the frontier, the main comparison of chapters 7 and 8 (section
4.6), ran before the fix, and its rerun is open work (section 10.1).

### One ranking for every `k`

Mann-Whitney, mutual information, correlation, Lasso and tree importance
rank all features in one fit that does not depend on `k`. So their top
10, top 50 and top 250 are cut from one ranking per split. Each cut
equals a separate fit at that `k`. In the method comparisons Lasso takes
no `k`; its ranking is used in the PAM50 recovery (section 7.6).

RFE needs one more step. It is fit once, down to the smallest `k`. With a
fixed step, a fit to a larger `k` runs the same rounds and stops earlier.
So we rank the features by the round they were removed in, and within a
round by their weight. This ranking gives RFE's selection at every larger
`k`. Tests check that every cut equals a separate fit. Chapter 6 shows the
time this saves.

## 4.3 Why we score genes with the U test, not the F-test

**The data.** Only the order of a gene's values is meaningful, and about
half the values sit tied at a floor (section 2.1).

**The F-test** compares class means. It depends on the size of the values,
so another, equally valid transform would change it. Its p-values assume
normal values with equal spread in both classes. The F statistic still
ranks genes sensibly. But Higher Criticism reads its tiny p-values, and
those rest on an assumption the data breaks.

**The Mann-Whitney U test** (Mann and Whitney 1947) uses only the order of
the values and assumes no normal curve. For a binary task, U divided by
n₁n₂ is exactly the AUC of the gene used alone as a score (Hanley and
McNeil 1982). So ranking genes by U-test p-value ranks them by how well
each one alone separates the classes, in either direction. Only the tie
correction, which differs from gene to gene, can shift this order
slightly.

Its p-value answers the question we want to ask: how likely is a
separation this good if the labels were shuffled, with the class sizes
fixed? Ties get average ranks, so the floor counts as one large group of
tied values. With two classes, the U
test is the same test as Kruskal-Wallis, "ANOVA on ranks". It needs
exactly two classes, and every task in the project is binary.

The p-value is the normal approximation with tie and continuity
corrections, as in SciPy's `mannwhitneyu(method='asymptotic')`.

Two other tests were considered. Welch's t-test allows unequal spread but
still assumes normal values. A permutation test assumes nothing, but its
smallest p-value is about 1 over the number of shuffles, far too coarse
for Higher Criticism.

So every filter in the experiments on real data that runs a statistical
test uses the U test. The one exception is FDR in the simulation of
section 8.11, whose features are normal. Correlation and mutual information, used in the frontier, run no
test. With one test for all filters, the experiments compare selection
rules, not tests. The F-test selectors stay in the package as drop-in
replacements for scikit-learn's, but the experiments do not use them.

**Its cost.** Before the speed work of chapter 6, a U-test fit took about
10 s on SCAN-B, against about 1 s for the F-test. Mann-Whitney ranks once
per split, so the cost is paid once per split, not once per `k`.

**Its limit.** The normal approximation is not exact in the extreme tail,
where HC2004 (section 4.4) reads its p-values. The exact permutation p-value would fix
this. But with ties it depends on each gene's own pattern of ties, so it
cannot be computed once for all genes.

**Numerical care.** Many genes have p-values below 10⁻³⁰⁸, where a
standard 64-bit float rounds to 0. Higher Criticism needs to tell them
apart. We compute the logarithm of the p-value directly from the normal
tail, so nothing underflows ([`featsel/selectors/log_pvalue/`](../featsel/selectors/log_pvalue/)).
On `scanb_small`, an earlier course copy of SCAN-B, 167 genes had an
F-test p-value of exactly 0 for the five classes. The module's tutorial
uses 2,000 synthetic features with 1,500 samples per class, 300 of them
with signal. There SciPy gives p = 0 to 170 features with the U test, and
to 189 with the F-test. The logs keep all of them distinct. Against
50-digit arithmetic (mpmath), the relative error of the log is at most
5 × 10⁻¹⁶ for the U test and 7 × 10⁻¹² for the F-test.

**A better model, not built.** An exact test that models the floor is
described in
[Appendix B.4](appendix_b_data_details.md#b4-a-better-model-for-floor-heavy-genes-not-built).

## 4.4 Higher Criticism

Most filters need `k` from the user. Higher Criticism (HC) picks its own
count. It comes from detecting rare, weak signals among many tests (Donoho
and Jin 2004, 2008).

**The idea.** Sort the N p-values: p₍₁₎ ≤ p₍₂₎ ≤ … ≤ p₍N₎. If no gene
carried signal, the p-values would be uniform, and p₍ᵢ₎ would sit near
its mean, μᵢ = i/(N+1) (Moscovich et al. 2016, Eq. 2.8). HC measures, at each i, how far the sorted p-values fall below
that line, in units of their standard deviation. Where this gap is
largest, the signal stands out most. HC keeps every feature up to that
point.

We implement two forms, HC2004 from the 2004 paper and HC2008 from the
2008 paper. The two papers divide by different terms:

| Variant | Statistic | Reference |
|---|---|---|
| HC2004 | HC(i) = √N · (μᵢ − p₍ᵢ₎) / √(p₍ᵢ₎(1 − p₍ᵢ₎)) | Donoho and Jin (2004), section 1.2 |
| HC2008 | HC(i) = √N · (μᵢ − p₍ᵢ₎) / √(μᵢ(1 − μᵢ)) | HC thresholding, Donoho and Jin (2008, 2009) |

HC2004's denominator is the p-value itself. At the smallest p-value, far below
1/N, HC(1) ≈ 1/√(N · p₍₁₎), which is very large. So HC2004 can peak at
the single strongest p-value. HC2008 divides by μᵢ, the expected
position, and does not.

HC2008 could also use the exact variance of the sorted p-value,
μᵢ(1 − μᵢ)/(N + 2). That would only multiply every HC(i) by the same
constant, √((N + 2)/N). The selected genes would not change, so this is
not a separate variant.

Donoho and Jin (2004) also give HC⁺, which skips p-values below 1/N. It
was not used. On `scanb_small`, the earlier course copy, every p-value
inside the search lay below 1/N, so HC⁺ would have had nothing to search.

The maximum is searched over the smallest alpha0 · N p-values. The 2008
paper gives alpha0 = 0.1 as an example ("e.g., α0 = 1/10", p. 14791); the
2009 paper "always use[s] α0 = .10" (just after Definition 1.2). The 2004 paper leaves
alpha0 open. We
use 0.1 for both forms.

**When HC is expected to work.** Its guarantees hold when the useful
features are *rare* (a small share of all features), *weak*, and
*independent*. Donoho and Jin (2004) built HC to detect this "rare and
weak" setting. Donoho and Jin (2008, 2009) turned it into a selection
threshold and proved it optimal there. Both papers derive the null
behavior of the sorted p-values from independent p-values. Correlated
genes give correlated p-values. Chapter 2 showed that none of our three
datasets meets these conditions. Sections 7.5 and 8.5 show what HC does then.

**Decision: alpha0 stays 0.1.** On SCAN-B, HC2008 stops at the edge of
its search on every split (section 7.5). We could raise alpha0, since we know the signal is
widespread. But that would tune the method to the answer. A selector must
not assume what it is meant to find.

## 4.5 Berk-Jones

Berk-Jones is a close relative of HC. It reads the same sorted p-values
but judges each one differently.

**The idea.** Under the null, the i-th smallest of N uniform p-values
follows a Beta(i, N − i + 1) distribution. HC summarizes that distribution
by a mean and a standard deviation. Berk-Jones uses all of it. It scores
rank i by the chance, under the null, of an i-th smallest p-value at least
this small:

BJ(i) = P(Beta(i, N − i + 1) ≤ p₍ᵢ₎)

A small BJ(i) means that p₍ᵢ₎ is surprisingly small for its rank. The
selector keeps every feature up to the rank where BJ(i) is smallest. This
minimum is the one-sided exact Berk-Jones statistic, M⁺ₙ, of Moscovich,
Nadler and Spiegelman (2016), after Berk and Jones (1979). It is searched
over the same smallest alpha0 · N = 10% of p-values as HC, so the two
compare on equal terms.

**Published as a test, used here as a selector.** The papers use the
statistic to test whether any signal is present. Using the rank of its
minimum as a threshold follows Donoho and Jin's step from HC as a test to
HC as a threshold. That step is ours for Berk-Jones, and its guarantees
are not proved.

**Computing it.** The strongest p-values are far below 10⁻³⁰⁸, and BJ(i)
is smaller still. So BJ is computed as a logarithm. Where the CDF is at
least 10⁻³⁰⁰, SciPy computes it directly. Below that, a series for the
incomplete beta function (DLMF 8.17.8) gives its logarithm. Tests check
it against 50-digit arithmetic.

**What to expect.** BJ judges each rank on its own null distribution. So
a single p-value far below the rest can win at rank 1, as in HC2004. Its
assumptions are those of HC: rare, weak and independent signal.

## 4.6 What the experiments used

| Experiment | Methods |
|---|---|
| The frontier (chapters 7 and 8) | Random, Mann-Whitney, HC2004, HC2008, Berk-Jones, correlation (Spearman), mutual information, Lasso, tree importance, RFE, RFECV, Sequential; chains of Mann-Whitney then Lasso, tree importance or RFE. The first frontier run had two solvers each for Lasso (liblinear and PPDNA) and RFE (liblinear and Newton); the configurations now run only PPDNA and Newton. |
| The classifier study (section 7.7) | Random and Mann-Whitney, with five classifiers |
| The unseen dataset (section 8.9) | Random, Mann-Whitney, HC2004, HC2008, Berk-Jones, correlation (Spearman), mutual information, Lasso (PPDNA), tree importance, RFE (Newton), RFECV (Newton, scored by accuracy and by AUC, section 7.8), Sequential (a batched Newton solver, section 6.2); chains of Mann-Whitney then Lasso, tree importance or RFE |
| Simulated rare signal (section 8.11) | Random, Mann-Whitney, HC2004, HC2008, Berk-Jones, FDR, Lasso (PPDNA) |
| The chains' first step (section 8.7) | the three chains, with Mann-Whitney keeping 250 or 1,000 features, on the image data |

Variance threshold, ANOVA, Percentile and FWE are in the package and
tested, but not in the experiments; FDR only in the simulation. ANOVA,
Percentile, FWE and FDR use the F-test, which the U test replaces on real
data (section 4.3). For the same reason, the correlation selector uses
Spearman (rank) correlation: Pearson's value changes with the transform
of the gene values, and only their order is meaningful (section 2.1). Variance threshold uses no label at
all.

## 4.7 Cost in time and memory

The measured times are in chapter 6 and section 7.11. This section gives
the cost that follows from each algorithm, as implemented in `featsel`.
It shows how each cost grows with the data. It does not give the constant
factors, and it does not include the GPU.

The symbols:

- n: training samples. p: features. k: features kept.
- T: trees in the forest. F: folds of an inner cross-validation.
- R: rounds of RFE (about 10 at step 0.1). t: Newton steps.
- m: the correlation selector's candidates, the features whose
  correlation with the label passes its threshold.
- r: the genes active in one Lasso Newton step.

The data itself takes O(np) memory for every method. The memory column
gives what a method needs beyond it.

| Method | Time per fit | Memory beyond the data |
|---|---|---|
| Random | O(p) | O(p) |
| Variance threshold, F-test (ANOVA, FWE) | O(np) | O(p) |
| FDR, Percentile | O(np + p log p): the F-test, then a sort of the p-values | O(p) |
| Mann-Whitney (U test) | O(p · n log n): one sort per feature | up to O(np) for the ranks |
| HC2004, HC2008 | the U test, plus O(p log p) to sort the p-values | as the U test |
| Berk-Jones | as HC, plus a log Beta CDF at each of the 0.1 · p searched ranks | as the U test |
| Correlation (Spearman) | O(p · n log n) for the ranks, plus O(n m²) for the pairs among the candidates | O(nm) for the candidates, plus one block of 1,024 × m correlations |
| Mutual information | O(p · n log n): per feature, one sort per class and one binary search per sample | O(n) per feature |
| Lasso (PPDNA) | per Newton step, O(np) for the products with X, plus O(n r² + min(n, r)³) for its linear system | O(n r + min(n, r)²) |
| Tree importance | O(T · √p · n log² n), if each tree has depth O(log n) | O(T n) for the trees |
| RFE (Newton) | O(n² p) for the n × n kernel X Xᵀ (below) and its updates, plus O(t n³) per round, R rounds | O(n²) |
| RFECV | about F + 1 times RFE | O(n²) |
| Sequential (Newton) | k steps. Step j fits about p · F models on j features, so O(t · F · p · n · k³) in total | O(p k²) for the Hessian blocks of all candidates |

How the derivations go:

- **The U test** sorts each feature once. The sort gives the ranks and the
  tie groups (section 6.2).
- **The forest** tries √p features at each node. Each level of a tree
  sorts at most n samples on them. If a tree has depth O(log n), that
  gives O(√p · n log² n) per tree. A fully grown tree on imbalanced data
  can be deeper, and then it costs more.
- **RFE** forms the kernel X Xᵀ once, in O(n² p). Each round subtracts the
  share of the removed features. Over all rounds this also costs O(n² p).
  Each Newton step solves an n × n system, in O(n³).
- **Lasso** solves its Newton system through the r active genes: an r × r
  system when r < n, else an n × n one, by the Woodbury identity
  ([`ppdna.py`](../featsel/selectors/lasso/ppdna.py)).
- **Sequential** at step j scores every remaining feature with j features
  on F folds. Each fit needs a j × j Hessian, in O(n j²) per Newton step.
  Summed over the k steps, this gives the k³. This is a bound for large
  k. At k = 5 and 10, the measured time grew about in step with k
  (section 7.10). At so small a k, work that does not grow with j, such
  as the passes over the data, probably dominates.

Choosing `k` or Lasso's C on validation sets adds work (section 3.11).

**What the costs predict.** The filters grow as p · n log n or less, so
they stay cheap at any p. RFE and RFECV grow with n² p and n³: they gain
when p ≫ n, and lose when samples far outnumber features (section 6.2). Sequential grows with
p and at least in step with k, so it runs only at small `k`
(section 3.11).
