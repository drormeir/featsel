# Choosing Few Genes from Many: Feature Selection for High-Dimensional Data

## Summary

**Problem.** Gene expression data describe each patient by tens of
thousands of genes, but a study has only hundreds or a few thousand
patients. A classifier trained on all the genes is slow and can learn
noise. Feature selection keeps a small set of features that still
predicts well. Many methods exist. It is not clear which one to use for a
given number of features, or how much better it is than random features.

**What we built.** `featsel`, a Python package of 16 selection methods
that drop into scikit-learn pipelines, and a runner that compares them
under one protocol from one configuration file: 16 random splits,
selection inside each split, the number of features chosen on validation
sets, a random control at every number of features, and paired tests
corrected for the overlap between splits. GPU paths make the slow methods
faster; exact solvers make them repeatable.

**Data.** Breast cancer gene expression (SCAN-B: 2,976 patients, 28,541
genes, five PAM50 subtype tasks), ResNet-50 embeddings of breast
ultrasound images (BreastMNIST: 780 images, 2,048 features), and aircraft
photos (FGVC-Aircraft: 1,200 photos), a dataset not seen during the
design.

**Results.** Selection helps most when few features are allowed: at 10
features, the best methods beat random features by 0.11 to 0.20 AUC on
SCAN-B and by 0.10 to 0.16 on the image data. As the number grows, random
features catch up, because thousands of correlated features carry the
signal. At the number each method chose, the top methods cannot be told
apart: on SCAN-B, eleven lie within 0.01 mean AUC. The Mann-Whitney U
test, which scores one feature at a time, is within 0.01 AUC of the best
on all three datasets, in under a second per fit on a GPU. Selection
rarely beats all the features: never on the image data, and on one SCAN-B
task of five. Data checks found repeated patients in SCAN-B and copied
images in BreastMNIST, which would leak between the training and the test
set.

**Conclusions.** On data like these, a simple test of one feature at a
time is almost as good as the complex methods, at a fraction of the cost.
The best method for a given budget cannot be known in advance, but
Mann-Whitney is a safe default. Many different feature sets predict
equally well, so stability must be reported next to accuracy.

## 1. Introduction

### 1.1 The problem

A breast tumour sample can be described by about 30,000 gene
measurements. A clinical study usually has hundreds to a few thousand
patients: our data has 2,976 patients and 28,541 genes. This setting, more features than
samples (p ≫ n), also appears in text and in image embeddings. It is hard
for four reasons. With more features than samples, a linear classifier can
separate any labelling of the training set, even a random one. If no gene
were related to the label, testing 28,541 genes at the 5% level would
still flag about 1,400 by chance. Genes work in groups, so many different
gene sets predict equally well. And methods that train a classifier for
each candidate feature can take hours.

Feature selection gives the classifier only the features that carry
information about the label. The model can then predict better, cost
less, and be read: a short gene list can be tested in a laboratory.

### 1.2 Tasks and questions

The project specification (`references/MLDS_Feature_Selection.pdf`) asks
for four tasks: (1) a feature selection pipeline for high-dimensional
data; (2) a comparison of selection techniques; (3) infrastructure that
speeds up the process; and (4) a run on a second dataset, not necessarily
gene expression.

The main question: **for a budget of `k` features, which method gives the
best classifier, and by how much does it beat `k` random features?** If
random features do almost as well, a method has found little. The second
question: **which methods give the best quality for their time?**

### 1.3 Earlier work

Comparative studies on gene expression found that simple classifiers did
as well as complex ones (Dudoit et al. 2002), or that multiclass SVMs
were the most accurate (Statnikov et al. 2005). Two reviews survey the
selection methods used in bioinformatics (Saeys et al. 2007) and on
microarray data (Bolón-Canedo et al. 2014). Most of their datasets
had tens to a few hundred samples. Four known pitfalls
shape our protocol (Table 1):

*Table 1. Known pitfalls and our answers.*

| Pitfall | Source | Our answer |
|---|---|---|
| Selecting genes on all the data, then cross-validating | Ambroise and McLachlan (2002) | Selection inside every split (section 3) |
| No random control | Neekhra et al. (2025) | Random features at every `k` (section 3) |
| Trusting the spread of k-fold scores | Bengio and Grandvalet (2004) | Repeated random splits and a corrected test (section 3) |
| Ignoring which features are chosen | Nogueira et al. (2018) | Stability for every method (section 3) |

### 1.4 What this project adds

1. **A comparison at a cohort size earlier studies lacked.** 2,976
   patients give test sets of 893, and a paired test corrected for
   overlapping splits checks every difference (section 3).
2. **Higher Criticism and Berk-Jones as selectors on real data.** Both
   thresholds were built for rare signal. On widespread signal, HC2008
   and Berk-Jones stopped at the edge of their search and acted as
   top-10% filters; HC2004 kept one to three features. On simulated rare
   signal, HC2008 chose a good count (sections 4, 6.4).
3. **A cost-quality frontier**: selection time against AUC for every
   method and setting, on each dataset (section 6.8).
4. **Drop-in selectors that are tested, fast and exact.** Those that copy
   a scikit-learn method select the same features, on the CPU and on a
   GPU. RFE, RFECV and Lasso reach the exact solution that scikit-learn's
   default solver stops short of (sections 4, 5).
5. **Checks of the data that change its use**: repeated patients and a
   computed label in SCAN-B, and copied images in BreastMNIST (sections
   2.1, 2.2).

This report keeps the main claims and their evidence. A detailed report
in the repository holds every task, every `k`, every test and every
design decision. A reference such as "D 7.2" points to its section 7.2,
in the folder `detailed_report/` of the public repository
[github.com/drormeir/featsel](https://github.com/drormeir/featsel).

## 2. Data

Three datasets have more features than samples (Table 2). SCAN-B is the
main dataset. BreastMNIST answers task 4 with another domain. FGVC-Aircraft
was added after the design was fixed, to test the pipeline on data it was
not built for. Each dataset has a loader that downloads the published
files once, checks them, and records every choice made on them (D 5.3).
The facts below are in `results/data_facts.csv` (D 2).

*Table 2. The three datasets.*

| | SCAN-B | BreastMNIST | FGVC-Aircraft |
|---|---|---|---|
| Domain | breast tumour RNA sequencing | breast ultrasound images | aircraft photos |
| Samples | 2,976 patients | 780 images | 1,200 photos |
| Features | 28,541 genes | 2,048 ResNet-50 embedding dimensions | 2,048 ResNet-50 embedding dimensions |
| Tasks | 5: each PAM50 subtype against the rest | 1: malignant (210) against benign or normal (570) | 1: Boeing 737 (800) against the Airbus A320 family (400) |
| Label | computed from 50 known genes | from the image source | the aircraft model |

### 2.1 SCAN-B

SCAN-B sequenced the tumours of thousands of breast cancer patients. The
data is public as GEO accession GSE96058 (Brueffer et al. 2018): 3,273
patients, 3,409 samples and 30,865 genes. We keep the 3,069 samples from
one sequencer, the HiSeq 2000, because two sequencers can differ for
technical reasons alone. We keep one sample per patient: 93 patients have
a tumour sequenced twice, and split between the training and the test
set, such a pair tests a model on a patient it has seen. We drop genes
with a single value and 11 nearly flat genes. This leaves 2,976 patients
and 28,541 genes. The course's copies of the data kept the repeated
patients, and the smaller copy lacked 11 of the 50 genes that the label
is computed from (below; D 2.1), so we load GSE96058 directly.

**The values.** Each value is log2(FPKM + 0.1), a transform of a read
count. Any other increasing transform would be equally valid, so only the
order of a gene's values is meaningful. An undetected gene sits at a
floor, log2(0.1): in the first 200 genes, 51% of the values are at it,
and 98.1% of all genes fail a test of normality. So we score genes with
the Mann-Whitney U test, which uses only the order, not with the F-test
(section 4).

**The label.** PAM50 (Parker et al. 2009) assigns each tumour one of five
subtypes: Basal, Her2, LumA, LumB or Normal. It gives the subtype of the
nearest of five reference profiles over 50 genes, and those 50 genes are
in our data. So predicting PAM50 partly re-learns a published classifier,
and scores may be higher than on a measured outcome. In return, we know
the right genes and can test whether a method finds them (section 6.5).

**Five binary tasks.** Every task is binary: each subtype against the
other four. The tasks are imbalanced. On Normal against the rest (200
against 2,776 patients), a model that always answers "rest" is right 93%
of the time, so plain accuracy is not the main score (section 3).

**The signal.** Genes work in groups, so many genes can carry the same
signal, and different gene sets can predict equally well. Sections 6.1,
6.3, 6.4 and 6.5 show the effects.

### 2.2 BreastMNIST

BreastMNIST (Yang et al. 2023) holds 780 breast ultrasound images
(Al-Dhabyani et al. 2020); we use its 224 × 224 release, MedMNIST+ (Yang
et al. 2024). Each image goes through ResNet-50
(He et al. 2016), trained on ImageNet, with its last layer removed. The
result is 2,048 numbers per image: more features than images. The signal
is again widespread: 24% to 28% of the features pass the U test (section
4) at a 5% false discovery rate: the expected share of false features
among those that pass is at most 5%. Unlike genes, pairs of features are only weakly
correlated (median absolute correlation 0.056).

**Copied images.** A published letter found copies in the source images
(Pawłowska et al. 2023): the same scan under a small geometric change (an
affine transform). Like
SCAN-B's repeated patients, a copy can fall in the training set and its
twin in the test set. Our loader finds them among BreastMNIST's own
images, by the letter's method: 150 groups of 339 images. A run without
them keeps 585 images (section 6.7).

### 2.3 FGVC-Aircraft

FGVC-Aircraft (Maji et al. 2013) holds 10,000 aircraft photos. The task
is Boeing 737 against the Airbus A320 family, two families of airliners
that look alike. Each photo becomes 2,048 numbers through ResNet-50, as
for BreastMNIST, but with torchvision's older ImageNet weights, V1. With
the V2 weights used for BreastMNIST, 80% of the features were exactly zero
and all the features reached only 0.742 AUC (the score of section 3); V1 gave 0.1% zeros and 0.862
(`results/aircraft_weights.csv`). As in the other two datasets, the
signal is widespread (D 2.4). The photos are for non-commercial
research only and are not stored in the repository.

## 3. Protocol

A selection method is judged by what its features do for a classifier on
samples it has not seen. Every method comparison follows one protocol
(Table 3; D 3).

*Table 3. The protocol.*

| Part | Choice |
|---|---|
| Splits | 16 random stratified splits; 70% of the samples in the training set |
| Selection | inside each split, on the training set only |
| Main score | ROC AUC on the test set, mean over the splits |
| Control | random features at every `k` |
| `k` | 10, 50, 250, 500, 1,000 (and 1,500, 2,000 on SCAN-B), chosen on validation sets |
| Classifier | logistic regression, L2 penalty, C fixed at 1 |
| Tests | the corrected resampled t-test, paired over splits |
| Stability | Nogueira's measure over the 16 splits |
| Timing | each selection fit 10 times (3 on FGVC-Aircraft; Sequential selection, section 4, once on the other two); the median is recorded |

**Selection inside each split.** On each split, the runner standardizes
the features on the training set, chooses `k`, fits the selector and then
the classifier, all on the training set. The test set is scored once.
Selecting features on all the data first is a known error (Ambroise and
McLachlan 2002): the test samples then help choose the features. On
BreastMNIST with shuffled labels, where nothing can be learned, the error
raises the median AUC over 20 splits from 0.465 to 0.645 (D 3.2).

**Random splits, not k-fold.** In 5-fold cross-validation, any two
training sets share three quarters of their samples, so the five scores
understate the real variation (Bengio and Grandvalet 2004). Each of our
splits draws a fresh stratified training set. On SCAN-B one split serves
all five tasks, so they are scored on the same patients.

**Why 16 splits.** Random splits still overlap, so we use the corrected
resampled t-test of Nadeau and Bengio (2003). Let d be the mean of the
differences in test AUC between two methods over J splits, and s² their
sample variance. Then

t = d / √((1/J + n~test~/n~train~) s²), with J − 1 degrees of freedom.

A difference is *established* when this test gives p < 0.05.
The plain paired t-test has 1/J alone. The second term does not shrink
with J. From 5 to 16 splits,
the 95% interval narrows by about a third; beyond 16 it can narrow by at
most 14% more. A difference not established at 16 splits needs more
samples, not more splits.

**The random control.** At every `k`, a selector that picks `k` features
at random gets the same splits and classifier as every method. A method
has value at `k` only if it beats random features at that `k`. Small
random subsets have matched selected ones on many high-dimensional
datasets (Neekhra et al. 2025), so the control is needed.

**ROC AUC.** The tasks are imbalanced, and AUC needs no decision
threshold. Macro-F1 and balanced accuracy were also recorded; they give
almost the same order of methods (D 7.12, 8.10).

**Choosing `k`.** Choosing `k` by test AUC would use the test set twice.
So the training set is cut into 5 folds. Each fold in turn is the
validation set: the selector is fit on the other four folds, and for each
`k` the classifier is scored on the held-out fold. The `k` with the best
mean validation AUC wins. Lasso chooses its penalty the same way, and
Higher Criticism and Berk-Jones choose their own count (section 4).

**Stability.** A method can score well and pick different features on
each split; its list is then an accident of the split. Nogueira's measure
(Nogueira et al. 2018) is 1 for identical sets and 0 for chance overlap,
and allows sets of different sizes. Over M splits, let p~f~ be the share
of splits that keep feature f, k the mean number of features kept, and N
the number of features:

Φ = 1 − [(1/N) Σ~f~ M/(M − 1) p~f~(1 − p~f~)] / [(k/N)(1 − k/N)].

## 4. Methods

Feature selection methods fall into three families (Guyon and Elisseeff
2003). We add two filters that choose their own count,
and the random control. Table 4 lists the methods the experiments used;
`featsel` also holds the F-test selectors of scikit-learn and a variance
filter (D 4.2).

*Table 4. The methods in the experiments.*

| Method | Family | Idea |
|---|---|---|
| Random | control | Pick `k` features at random. |
| Mann-Whitney | filter | Rank features by the U test (below). |
| Correlation | filter | Rank by Spearman correlation with the label; drop one of each highly correlated pair. |
| Mutual information | filter | Rank by mutual information, which catches any dependence. |
| HC2004, HC2008 | filter, own count | Keep the features below the Higher Criticism threshold. |
| Berk-Jones | filter, own count | Keep the features below the Berk-Jones threshold. |
| Lasso | embedded | L1-penalized logistic regression; keep the non-zero weights. |
| Tree importance | embedded | Rank by a random forest's impurity importance. |
| RFE | wrapper | Train a linear SVM, remove the weakest features, repeat (Guyon et al. 2002); we remove 10% per step. |
| RFECV | wrapper | RFE, with the count chosen by inner cross-validation, scored by accuracy (its default) or by AUC (section 6.6). |
| Sequential | wrapper | Add the feature that most improves cross-validated accuracy; repeat. Run at `k` = 5 and 10 only: one fit takes about 10 minutes on SCAN-B (section 6.6). |
| Chains | two stages | Mann-Whitney keeps 1,000 genes (500 features on the image data); Lasso, tree importance or RFE picks from them. RFE alone is weak at small `k`, and the filter first helps it. The first count is fixed, not tuned (section 6.6). |

**Why the U test.** The F-test compares class means and assumes normal
values. Gene values are not normal, and only their order is meaningful
(section 2.1). The Mann-Whitney U test (Mann and Whitney 1947) uses only
the order. For a binary task, U divided by n₁n₂ is the AUC of the feature
used alone (Hanley and McNeil 1982), so ranking by the U test ranks
features by how well each alone separates the classes. Every filter of
Table 4 that runs a test uses the U test, so the experiments compare selection rules,
not tests. Many genes have p-values below 10⁻³⁰⁸, where a 64-bit float
rounds to zero. We compute the logarithm of the p-value directly, so
Higher Criticism can still tell them apart (D 4.3).

**Why Higher Criticism and Berk-Jones.** Most filters need `k` from the
user, and choosing `k` on validation sets costs more than the selection
itself (section 6.8). Higher Criticism and Berk-Jones choose their own
count from the p-values of all the features, so they skip that search.

**Higher Criticism.** Sort the N p-values, p~(1)~ ≤ … ≤ p~(N)~. If no
feature carried signal, p~(i)~ would sit near its mean, i/(N + 1)
(Moscovich et al. 2016). Higher Criticism measures, at each i, how far
p~(i)~ falls below that line, in standard deviations, and keeps every
feature up to the largest gap (Donoho and Jin 2004, 2008). HC2004 is the
2004 paper's form:

HC(i) = √N (i/N − p~(i)~) / √(p~(i)~ (1 − p~(i)~)).

HC2008 is the 2008 paper's form with one change: with μ~i~ = i/(N + 1)
in place of i/N,

HC(i) = √N (μ~i~ − p~(i)~) / √(μ~i~ (1 − μ~i~)).

Both search the smallest 10% of
p-values, the example value of the 2008 paper. We kept this limit, though the signal here is widespread: widening it to fit
the data would tune the method to the answer.

**Berk-Jones.** Under the null, the i-th smallest of N uniform p-values
follows a Beta(i, N − i + 1) distribution. Berk-Jones scores each rank by
the chance of a p-value at least this small at that rank,

BJ(i) = P(Beta(i, N − i + 1) ≤ p~(i)~),

and keeps the features up to the rank, in the same smallest 10%, where
that chance is smallest (Berk and Jones
1979; Moscovich et al. 2016). The papers use it as a test of whether any
signal exists; using its minimum as a threshold is our step, and its
guarantees are not proved.

**Rare signal.** Both thresholds were built for signal that is rare, weak
and independent. In none of the three datasets is the signal rare
(section 2). Section 6.4 shows what the thresholds do then, and on
simulated rare signal.

## 5. Software and speed

`featsel` serves two kinds of use (D 5).

1. **Drop-in selectors.** Every selector is a scikit-learn estimator. It
   replaces a scikit-learn selector in an existing pipeline with no other
   change, and works with `clone`, `cross_val_score` and `GridSearchCV`.
   Each selector that copies a scikit-learn method is tested to select the
   same features and, except Lasso, to be at most 5% slower. Lasso has no
   speed test, because no scikit-learn solver reaches its exact solution
   (below). Each GPU path
   is tested to select the same features as the CPU path.
2. **A config-driven runner.** One YAML file names the dataset, the
   selectors, the values of `k` and the classifier. One command runs the
   whole protocol of section 3 (Appendix A) and writes one CSV row per
   task, split, method and `k`, with a JSON record of the machine,
   the library versions and the git commit.

```python
pipe = make_pipeline(StandardScaler(),
                     MannWhitneySelector(n_features=50),
                     LogisticRegression(max_iter=1000))
scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
```

**The runner.** A method that ranks all features in one fit, such as
Mann-Whitney, is fitted once per split, and every `k` is cut from its
ranking. Each split is prepared once and shared by all workers in memory.
Long runs on a 16 GB laptop crashed at first, so the runner caps each
worker's memory, skips a cell that does not fit, restarts with fewer
workers when one dies, writes each row as it finishes, and resumes a
stopped run. Every seed is fixed, so a rerun gives the same features and
scores, for any number of workers.

**Faster than plain scikit-learn (task 3).** One small grid ran with
scikit-learn's `cross_validate` and with the runner: SCAN-B's Basal task
on 3,000 random genes, 5 splits, and Lasso and RFE at three values of
`k`. In the median of three runs, the runner was 4.3 times faster on one
core and 10.6 times on 12 threads (3.9 to 4.5 and 10.1 to 10.8 over the
runs), with identical test scores (`results/speed_benchmark.csv`). It
does less work: it prepares each split once and ranks once per split. On
this grid, 12 workers gained 2.5 times over one. On all the genes and all five
tasks, an older run gained at most 1.8 times, from 3 workers on. There,
each selection fit ran 1.75 times slower beside others. A fit reads the
whole training set, so this points to the memory bandwidth that all cores
share (D 6.1).

**Each method, faster.** Each selector has a speed notebook that times it
against the scikit-learn original on synthetic data shaped like SCAN-B,
and checks that both pick the same features (D 6.2). On a GTX 1650 GPU,
the filters run 2.6 to 49 times faster than the original, with the same
features: mutual information 49 times, bit for bit, and the U test 7.7
times faster than our own NumPy version.
Sequential selection solves all candidate features of a step at once and
runs 34 times faster.

**Exact solvers.** scikit-learn's default solver for an L1 or linear-SVM
model, liblinear, stops at a tolerance of 10⁻⁴, before the exact solution.
With correlated genes, this changes which genes are kept. Unless its seed
is fixed, liblinear also shuffles the data differently on each fit and
keeps different genes (D 6.2). RFE and RFECV
now solve each SVM exactly by Newton's method (Chapelle 2007), and Lasso by PPDNA (Liu and Zhou 2023). With no
randomness, a refit gives the same genes. Newton's method works through
the samples' n × n kernel, so it pays when features outnumber samples.

**Where a GPU does not help.** The GTX 1650 runs 64-bit arithmetic at
1/32 of its 32-bit speed, and RFE needs 64 bits: with 32 bits it
selected different genes (D 6.2). Copying the data takes
about half the time of a fast filter. Tree importance runs on the CPU,
with all threads.

## 6. Results

Every result follows the protocol of section 3: 16 splits, 70% of the
samples in the training set, `k` chosen on validation sets, and ROC AUC on
the test set, as a mean over the splits. A difference is established
when the corrected test gives p < 0.05. The detailed report
gives every task, every `k` and every test (D 7, D 8, D C).

### 6.1 Selection against random features

Selection matters most when few features are allowed (Figure 1). At 10
genes on SCAN-B, every filter and tree importance beat random genes by
0.12 to 0.19 AUC, on every task. With the two-stage chains and Sequential
selection, the range is 0.11 to 0.20. At 10 features on the image data,
the filters and tree importance beat random features by 0.10 to 0.16. Each
of these gains is established and holds on all 16 splits (D 7.2, 8.2,
8.9).

As `k` grows, random features catch up. On Basal, no method beats random
genes from 250 genes on. On BreastMNIST, no gain is established at 1,000
features. On LumA and LumB, a gain of about 0.02 remains at 2,000 genes.
The reason is in the data: thousands of genes carry the signal, and they
move together (section 2). A large random set holds enough of them.

RFE alone is the exception at small `k` on SCAN-B. At 10 genes it scored
below random genes on LumA and Normal, though neither loss is
established (p = 0.30 and 0.06).

![Figure 1](figures/curves.png)

*Figure 1. Test AUC against the number of features kept, for one method
of each family and for random features. The gap to the gray dashed line
is what selection adds at that `k`. The dotted line is all the features.
Sources: D 7.2, 8.2.*

### 6.2 Selection against all features

On the image data, no method beats all 2,048 features. RFE ties them:
−0.001 AUC on BreastMNIST (p = 0.85) and −0.0003 on FGVC-Aircraft
(p = 0.92). On both, Lasso, correlation, the three chains, HC2008,
Berk-Jones and HC2004 lose to all
features (D 8.3, 8.9).

On SCAN-B, 20 of the 70 comparisons with all 28,541 genes are
established: 10 gains and 10 losses. All 10 gains are on one task, LumB:
ten methods gain 0.014 to 0.023 AUC, each on all 16 splits. On the other
four tasks no gain is established. HC2004 loses on four tasks and RFECV
scored by accuracy on three. Lasso and two chains lose on one task each.
The other filters and tree importance show no established loss (D 7.3).

So on these datasets, selection buys a smaller model more often than a
better one.

### 6.3 No single best method

At the `k` each method chose, the top methods cannot be told apart. On
SCAN-B, eleven methods lie within 0.01 of each other in mean AUC
(Table 5), and a different method leads on each task (D 7.1). On the
image data, RFE ties all features, and Mann-Whitney is within 0.01 of them
on both datasets.

The best method also changes with `k`. On BreastMNIST, Mann-Whitney beats
RFE by 0.066 at 10 features (p = 0.04), and RFE beats Mann-Whitney at 250
and 500 features (p = 0.049 and 0.006); at 1,000 they tie (p = 0.39).
FGVC-Aircraft shows the same turn (D 8.2, 8.9). A filter scores each feature alone; RFE looks
at the features together, and that pays only once many are kept.

Stability separates methods that AUC does not. Tree importance reaches
the same mean AUC as Mann-Whitney on SCAN-B, 0.969, but its stability is
0.27 against 0.76: it picks a different gene set on each split. So many
different gene sets predict equally well, and a gene list from one split
is not a finding by itself.

*Table 5. Each method at the `k` it chose on validation sets: test AUC on
each dataset, then SCAN-B's selection time and stability. SCAN-B
values are means over its five tasks; the time is the mean of each task's
median time per fit. Tree importance, RFE and RFECV ran on the CPU, also
as the second step of a chain; every other method ran on the GPU. On the
CPU alone, on synthetic data shaped like SCAN-B, the U-test methods
(Mann-Whitney, HC, Berk-Jones) take 3.8 to 3.9 s per fit, correlation
6.9 s, Lasso 25 s at one penalty, and mutual information 18 s on 10,000 of the
features (D 6.2). Over
the 16 splits, the interquartile range of a method's AUC is 0.006 to
0.010 on a SCAN-B task (HC2004 0.018, RFECV scored by accuracy 0.044)
and 0.017 to 0.069 on the image data. The tests are paired, so a smaller
difference can still be established. Without copies: section 6.7.
Source: `final_report/figures/table5.csv`.*

| Method | SCAN-B | BreastMNIST | BreastMNIST,<br>no copies | FGVC-<br>Aircraft | SCAN-B,<br>s per fit | SCAN-B<br>stability |
|---|---|---|---|---|---|---|
| Mutual information | 0.971 | 0.916 | 0.887 | 0.847 | 3.6 | 0.70 |
| Berk-Jones | 0.970 | 0.867 | 0.847 | 0.815 | 0.66 | 0.92 |
| HC2008 | 0.970 | 0.868 | 0.848 | 0.817 | 0.68 | 0.92 |
| Correlation (Spearman) | 0.969 | 0.905 | 0.866 | 0.824 | 3.6 | 0.78 |
| Mann-Whitney | 0.969 | 0.918 | 0.883 | 0.854 | 0.67 | 0.76 |
| Tree importance | 0.969 | 0.906 | 0.885 | 0.851 | 17 | 0.27 |
| Mann-Whitney, then Lasso | 0.968 | 0.878 | 0.858 | 0.828 | 0.79 | 0.46 |
| Mann-Whitney, then tree importance | 0.965 | 0.879 | 0.855 | 0.814 | 2.8 | 0.51 |
| Lasso | 0.964 | 0.898 | 0.882 | 0.846 | 0.59 | 0.36 |
| All features (no selection) | 0.963 | 0.924 | 0.895 | 0.862 | — | — |
| Mann-Whitney, then RFE | 0.962 | 0.889 | 0.860 | 0.829 | 3.0 | 0.42 |
| RFECV, scored by AUC | 0.962 | 0.912 | 0.884 | 0.858 | 5.7 | 0.23 |
| RFE | 0.961 | 0.923 | 0.891 | 0.862 | 6.9 | 0.33 |
| HC2004 | 0.874 | 0.758 | 0.771 | 0.674 | 0.68 | 0.67 |
| RFECV, scored by accuracy | 0.720 | 0.889 | 0.847 | 0.854 | 6.3 | 0.20 |

### 6.4 Higher Criticism and Berk-Jones

HC2008 kept 2,854 or 2,855 genes on every split and task of SCAN-B. That
is the edge of its search: 10% of 28,541 genes. On both image datasets it
kept 204 features, again the edge. Berk-Jones stopped at or just before
the same edge (D 7.5, 8.5). These thresholds were built for rare signal.
Here the signal is widespread, so they find no point where it ends, and
they act as a top-10% filter.

As such a filter, HC2008 is good on SCAN-B: 0.970 mean AUC, 0.001 below
the best, with the highest stability (0.92) and no `k` to choose. On
BreastMNIST, 204 features are too few: HC2008 loses 0.056 AUC to all
features (p < 0.001). HC2004 keeps one to three features. It is the
weakest method on both image datasets, and on SCAN-B only RFECV scored
by accuracy is weaker.

**Simulated rare signal** (D 8.11): 10,000 independent normal features, 40
of them moved by 0.4 standard deviations in the positive class. HC2008
kept a median of 34 features and had the best mean AUC, 0.858. Mann-Whitney
with a validated `k` reached 0.843; the difference is not established
(p = 0.62). Berk-Jones (0.813, 12 features) and an FDR filter, run only
here (the features that pass the F-test at a 5% false discovery rate,
section 2.2), were stricter, and
HC2004 kept one or two. All features gave 0.629.

### 6.5 Does selection find the PAM50 genes?

The PAM50 label is computed from 50 known genes (section 2). A side study
measured where these genes lie in each method's ranking, over 10 splits
(`results/pam50_recovery.md`). In their top 50 genes, the methods found on
average 12% of them (Mann-Whitney), 11% (tree importance), 9% (Lasso) and
3% (RFE). Mann-Whitney found half of the 50 by rank 1,031; random ranking
would need rank 14,271.

Yet at 50 genes, Mann-Whitney and tree importance predict the subtypes
well (Figure 1). A likely reason is that genes which move with the PAM50
genes carry the same signal. So a low share of PAM50 genes is not poor
selection. It shows again that a
gene list is one of many equally good answers. Mann-Whitney's lead here is
partly built in: Parker et al. (2009) chose the 50 genes by ranking each
gene alone, by a t-test for each subtype.

### 6.6 Wrappers need care

RFECV chooses its count by an inner cross-validation. With its default
inner score, accuracy, it kept a single gene on every split of Her2, LumB
and Normal, and scored 0.51 to 0.58 AUC. These tasks are imbalanced, and
accuracy rewards predicting the larger class. Scored by AUC, RFECV reached
0.962 mean AUC (D 7.8).

RFE alone is weak at 10 genes on SCAN-B: 0.78 mean AUC. A Mann-Whitney
filter first, keeping 1,000 genes, raised it to 0.95, the best of all
methods at 10 genes on four tasks of five. Sequential selection beat
random genes on every task at 10 genes, but this chain beat it on all
five tasks, in 3 s per fit instead of 10 minutes (D 7.9, 7.10).

A chain's first count was set by hand. On the image data, its 500 also
caps `k` at 250, and RFE and tree importance chose that cap. A side
study kept 250 or 1,000 instead, on the same splits. At 1,000, every
chain gained 0.012 to 0.025 AUC on BreastMNIST, with and without the
copies, and on FGVC-Aircraft. Only RFE's gains on BreastMNIST and
FGVC-Aircraft are established (p = 0.047 and 0.025); without the copies,
no gain is (D 8.7). The main results keep 500: choosing the count now by
test AUC would use the test set twice.

### 6.7 Without the copied images

BreastMNIST holds 150 groups of copied images (section 2). Without them,
585 images remain, and every method of Table 5 ran again. All features
fall from 0.924 to 0.895 AUC, and every method but HC2004 by 0.016 to
0.042; HC2004 rises by 0.013. The
leading methods change places within 0.009 AUC. RFE still ties all
features, and the chains, HC2008, Berk-Jones and HC2004 still lose to
them (D 8.8). So the copies inflate the scores but do not change the
conclusions.

### 6.8 Cost

The cheap end of Figure 2 wins. On SCAN-B, HC2008, Berk-Jones and
Mann-Whitney reach 0.969 to 0.970 mean AUC in under 0.7 s per fit on the
GPU. The best mean, mutual information at 0.971, takes 3.6 s, and tree
importance takes 17 s for 0.969 (Table 5). On the image data, every method
except Sequential selection runs in about a second per fit or less (tree
importance 1.1 s on FGVC-Aircraft), so cost does not decide the choice
there.

Choosing `k` costs more than selecting. For Mann-Whitney on SCAN-B, the
search on 5 validation sets takes about 28 s per split, against 0.67 s for
one fit (D C.1). A method
that chooses its own count, such as HC2008, skips this search.

![Figure 2](figures/frontier.png)

*Figure 2. Cost against quality. Each point is one method at one `k`:
its selection time per fit, timed as in Table 5, and its mean test AUC.
The dark line is the frontier: no point is both faster and better than a
point on it. SCAN-B is the mean of its five tasks. Sources: D 7.1, 8.1, 8.9.*

## 7. Discussion

### 7.1 The answer

For a budget of `k` features, which method gives the best classifier, and
by how much over random? At small `k`, the answer depends on the dataset.
At 10 genes on SCAN-B, Mann-Whitney then RFE is best on four tasks of
five, and the filters and tree importance beat random genes by 0.12 to
0.19 AUC. At 10
features on the image data, the filters and tree importance, alone or
chained, lead, by 0.10
to 0.16 over random. At the `k` each method chose, the top methods cannot
be told apart. If time also counts, a GPU filter is the practical choice:
on SCAN-B, HC2008 or Mann-Whitney reaches 0.970 or 0.969 mean AUC in under
a second per fit, 0.001 to 0.002 below the best.

### 7.2 What holds on every dataset, and what changes

On all three datasets, selection beats random features most at small `k`;
Mann-Whitney is within 0.01 AUC of the best, at the cheap end of the
frontier; and HC2008 stops at the edge of its search. Wherever RFECV ran
with both inner scores, AUC did better than accuracy. What changes is the best method for a given budget. At
large `k`, RFE leads on both image datasets but trails the filters on
SCAN-B (Figure 1). Whether selection beats all features also changes: never on the
image data, on one SCAN-B task of five. So the best method for a budget
cannot be known in advance from these datasets. What can be known is a
safe default.

### 7.3 Key decisions

*Table 6. The decisions that shaped the results (D 9.2).*

| Decision | Why | Effect |
|---|---|---|
| Select inside every split | Selection on all the data lets the test samples choose the features | With shuffled labels on BreastMNIST, the error raised AUC from 0.465 to 0.645 (section 3) |
| A random control at every `k` | A method has found something only if it beats random features of the same number | It showed where selection pays: at small `k` (section 6.1) |
| A test corrected for overlapping splits | Random splits share most of their training samples | Established differences from all genes fell from 52 to 20 of 70 comparisons (section 6.2; D 9.3) |
| GSE96058, one sample per patient, not the course copies | The copies kept 93 repeated patients, and the smaller one lacked 11 of the 50 PAM50 genes | It removed a leak and made the PAM50 test possible (sections 2.1, 6.5) |
| The U test for every filter | Only the order of gene values is meaningful, and many sit at a floor | One valid test, so filters differ only in their selection rule (sections 2.1, 4) |
| Find the copied images | Copies fall on both sides of a split | Scores fell by 0.016 to 0.042 AUC, except HC2004's; the conclusions held (section 6.7) |
| Exact solvers for RFE, RFECV and Lasso | liblinear stops before the solution, and unless seeded keeps different genes on each run | Repeatable selection, at the exact solution (section 5) |
| RFECV scored by AUC | Accuracy rewards the larger class | RFECV rose from 0.720 to 0.962 mean AUC on SCAN-B (Table 5, section 6.6) |
| Higher Criticism's search kept at 10% | Widening it to fit the data would tune the method to the answer | Its behaviour on widespread signal is reported as it is (section 6.4) |
| A third dataset, unseen during the design | Two datasets shaped the design, so they cannot test it | The pipeline ran unchanged on FGVC-Aircraft (section 6; D 8.9) |

### 7.4 Limits

- **Data.** Three datasets, two of them embeddings from the same network.
  Every SCAN-B task predicts PAM50, a label computed from the genes; no
  measured target was run. The main BreastMNIST results include the
  copied images; a run without them gives the same conclusions (section
  6.7).
- **Protocol.** One classifier in the main runs, logistic regression with
  C fixed at 1. A side study with older settings (12 splits, half the
  patients in the training set, macro-F1) added a linear SVM, a random
  forest, k-nearest neighbours and shrinkage LDA. With each of the five,
  Mann-Whitney beat random genes at 10 genes by 0.20 to 0.23 macro-F1, on
  every SCAN-B task and split (D 7.7). One training fraction and 16 splits; the spread over splits is not a
  confidence interval for new patients. The five SCAN-B tasks share
  patients, so they are not independent. Each selector's own settings are
  fixed, except Lasso's penalty; so is a chain's first count (section
  6.6). The correlation selector drops a gene
  whose only redundant partner was itself dropped (D 4.2).
- **Cost.** Every time comes from one laptop: Intel i7-9750H, 16 GB of
  memory, and an NVIDIA GTX 1650 with 4 GB. A time is meaningful only on
  that machine. The runner's speed-up comes from three runs of one small
  grid.
- **Higher Criticism.** No real dataset here has rare signal. One
  simulation, with independent features, tests it in that setting.

### 7.5 Open research questions

- **A real dataset with rare signal**, and simulations with other
  rarities, strengths and correlated features, would test further whether
  Higher Criticism and Berk-Jones choose a good count.
- **An exact test for floor-heavy genes.** A test that separates how often
  a gene is detected from how high it is when detected would fit these
  data better than the U test, which merges both (D B.4).
- **How often does a random set beat a chosen one?** For a set of `k`
  genes chosen by a method, the share of random size-`k` sets that predict
  at least as well on the same split is an empirical p-value for the
  chosen set. It would test a published gene signature against chance
  signatures of its size.
- **Stable selection.** Many methods pick different but equivalent genes
  on each split. Selecting groups of correlated genes, or keeping the
  genes chosen most often over repeated subsamples, could give stable
  lists.
- **A measured target**, such as the pathologist's estrogen receptor
  status, would test the conclusions against clinical truth. The loader
  already supports it.

## References

Al-Dhabyani, W., Gomaa, M., Khaled, H. and Fahmy, A. (2020). Dataset of
breast ultrasound images. *Data in Brief* 28, 104863.
https://doi.org/10.1016/j.dib.2019.104863

Ambroise, C. and McLachlan, G. J. (2002). Selection bias in gene extraction
on the basis of microarray gene-expression data. *PNAS* 99(10), 6562-6566.
https://doi.org/10.1073/pnas.102102699

Bengio, Y. and Grandvalet, Y. (2004). No unbiased estimator of the variance
of K-fold cross-validation. *Journal of Machine Learning Research* 5,
1089-1105.

Berk, R. H. and Jones, D. H. (1979). Goodness-of-fit test statistics that
dominate the Kolmogorov statistics. *Zeitschrift für
Wahrscheinlichkeitstheorie und verwandte Gebiete* 47, 47-59.
https://doi.org/10.1007/BF00533250

Bolón-Canedo, V., Sánchez-Maroño, N., Alonso-Betanzos, A., et al. (2014). A
review of microarray datasets and applied feature selection methods.
*Information Sciences* 282, 111-135.
https://doi.org/10.1016/j.ins.2014.05.042

Brueffer, C., Vallon-Christersson, J., Grabau, D., et al. (2018). Clinical
value of RNA sequencing-based classifiers for prediction of the five
conventional breast cancer biomarkers. *JCO Precision Oncology* 2, 1-18.
https://doi.org/10.1200/PO.17.00135. Data: GEO accession GSE96058.

Chapelle, O. (2007). Training a support vector machine in the primal.
*Neural Computation* 19(5), 1155-1178.
https://doi.org/10.1162/neco.2007.19.5.1155

Donoho, D. and Jin, J. (2004). Higher criticism for detecting sparse
heterogeneous mixtures. *Annals of Statistics* 32(3), 962-994.
https://doi.org/10.1214/009053604000000265

Donoho, D. and Jin, J. (2008). Higher criticism thresholding: optimal
feature selection when useful features are rare and weak. *PNAS* 105(39),
14790-14795. https://doi.org/10.1073/pnas.0807471105

Dudoit, S., Fridlyand, J. and Speed, T. P. (2002). Comparison of
discrimination methods for the classification of tumors using gene
expression data. *Journal of the American Statistical Association*
97(457), 77-87. https://doi.org/10.1198/016214502753479248

Guyon, I. and Elisseeff, A. (2003). An introduction to variable and feature
selection. *Journal of Machine Learning Research* 3, 1157-1182.

Guyon, I., Weston, J., Barnhill, S. and Vapnik, V. (2002). Gene selection
for cancer classification using support vector machines. *Machine Learning*
46(1-3), 389-422. https://doi.org/10.1023/A:1012487302797

Hanley, J. A. and McNeil, B. J. (1982). The meaning and use of the area
under a receiver operating characteristic (ROC) curve. *Radiology* 143(1),
29-36. https://doi.org/10.1148/radiology.143.1.7063747

He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for
image recognition. *CVPR*, 770-778. https://doi.org/10.1109/CVPR.2016.90

Liu, Y.-J. and Zhou, W. (2023). Dual Newton proximal point algorithm for
solution paths of the ℓ1-regularized logistic regression. Preprint.
https://arxiv.org/abs/2310.19353

Maji, S., Rahtu, E., Kannala, J., Blaschko, M. and Vedaldi, A. (2013).
Fine-grained visual classification of aircraft. Preprint.
https://arxiv.org/abs/1306.5151

Mann, H. B. and Whitney, D. R. (1947). On a test of whether one of two
random variables is stochastically larger than the other. *Annals of
Mathematical Statistics* 18(1), 50-60.
https://doi.org/10.1214/aoms/1177730491

Moscovich, A., Nadler, B. and Spiegelman, C. (2016). On the exact
Berk-Jones statistics and their p-value calculation. *Electronic Journal of
Statistics* 10(2), 2329-2354. https://doi.org/10.1214/16-EJS1172

Nadeau, C. and Bengio, Y. (2003). Inference for the generalization error.
*Machine Learning* 52(3), 239-281. https://doi.org/10.1023/A:1024068626366

Neekhra, B., Gupta, D. and Chakrabarti, P. P. (2025). On the
(in)significance of feature selection in high-dimensional datasets.
Preprint. https://arxiv.org/abs/2508.03593

Nogueira, S., Sechidis, K. and Brown, G. (2018). On the stability of
feature selection algorithms. *Journal of Machine Learning Research*
18(174), 1-54.

Parker, J. S., Mullins, M., Cheang, M. C. U., et al. (2009). Supervised risk
predictor of breast cancer based on intrinsic subtypes. *Journal of Clinical
Oncology* 27(8), 1160-1167. https://doi.org/10.1200/JCO.2008.18.1370

Pawłowska, A., Karwat, P. and Żołek, N. (2023). Letter to the Editor. Re:
"Dataset of breast ultrasound images". *Data in Brief* 48, 109247.
https://doi.org/10.1016/j.dib.2023.109247

Saeys, Y., Inza, I. and Larrañaga, P. (2007). A review of feature selection
techniques in bioinformatics. *Bioinformatics* 23(19), 2507-2517.
https://doi.org/10.1093/bioinformatics/btm344

Statnikov, A., Aliferis, C. F., Tsamardinos, I., et al. (2005). A
comprehensive evaluation of multicategory classification methods for
microarray gene expression cancer diagnosis. *Bioinformatics* 21(5),
631-643. https://doi.org/10.1093/bioinformatics/bti033

Yang, J., Shi, R., Wei, D., et al. (2023). MedMNIST v2 - a large-scale
lightweight benchmark for 2D and 3D biomedical image classification.
*Scientific Data* 10, 41. https://doi.org/10.1038/s41597-022-01721-8

Yang, J., Shi, R., Wei, D., et al. (2024). MedMNIST+: 18x standardized
datasets for 2D and 3D biomedical image classification with multiple size
options: 28 (MNIST-like), 64, 128, and 224. Zenodo, version 3.0.
https://doi.org/10.5281/zenodo.10519652

## Appendix A. How to reproduce

Every command runs from the repository root. Setup, with PyTorch for the
GPU paths and the image embeddings:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[gpu,embeddings]"
```

Each loader downloads its data on first use and caches it: GSE96058
(592 MB), BreastMNIST (31 MB), FGVC-Aircraft (2.75 GB). An experiment is
one command:

```bash
.venv/bin/python -m featsel.run \
    --config configs/frontier_scanb70_gpu.yaml --n-jobs 1
```

The configs behind section 6 are `configs/frontier_scanb70_*.yaml`,
`configs/frontier_breastmnist70_*.yaml`, `configs/aircraft70_gpu.yaml`, `configs/rare_signal70.yaml`
and, for the chains' first count, `configs/chain_size_*.yaml`.
Adding `--resume` continues a stopped run. Each run writes a CSV, and a
JSON file with the machine, the library versions and the git commit. D A
gives the command behind every summary table. This report's figures and
Table 5 come from three scripts, each with its exact command in its
docstring: `scripts/plot_final_frontier.py`, `scripts/plot_curves.py` and
`scripts/final_table.py`.

One seed, 42, fixes every random step; split `s` uses 42 + `s`. A rerun
gives the same features and scores, for any number of workers. The tests
run with `.venv/bin/python -m pytest`; GPU tests are skipped without a
GPU. On 2026-11-21, on the laptop of section 7.4 with its GPU, all 413
tests passed in 23 minutes (`results/test_run.md`).

## Appendix B. Engineering notes

**Where a selector runs.** Ten selectors share one device rule. With
`device='auto'`, the default, a fit uses the GPU when one is free and
falls back to the CPU otherwise, so a parallel run never waits. With
`'cuda'` it waits for the GPU; with `'cpu'` it never uses it. A file lock
lets one fit at a time hold the GPU. The experiments set `'cuda'`, so
every GPU time was taken on the GPU. Each result records the device it
used.

**The U test on a GPU.** One sort per gene gives its ranks and its groups
of tied values. On the GPU, PyTorch sorts thousands of genes at once and
sums the ranks exactly in 64-bit floats. The p-value is finished on the
CPU, so every device gives identical results.

**Mutual information, bit for bit.** scikit-learn estimates it from
nearest neighbours with a tree. For one feature, a sort per class finds
the k-th neighbour, and a binary search counts the points within its
radius. With scikit-learn's own random noise and order of sums, the
scores are identical to scikit-learn's.

**Lasso's solver.** PPDNA (Liu and Zhou 2023), a semismooth Newton method,
needed five changes from the paper on SCAN-B-shaped data (D D.3):

1. the intercept is penalized, as liblinear penalizes it;
2. the proximal step grows five-fold per outer step, up to a limit set by
   rounding;
3. the Newton matrix is factored in 32-bit floats, where the GPU is fast,
   and the solution is refined in 64 bits;
4. the inner loop also stops when full Newton steps no longer shrink the
   gradient;
5. the outer loop also stops when three outer steps fail to halve the best
   residual.
