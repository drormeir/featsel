# 1. The problem

[Back to contents](README.md) · Next: [2. The data](02_the_data.md)

## 1.1 What feature selection is

A dataset describes samples by features. Here the samples are patients or
images. The features are genes, or numbers that describe an image. A
classifier learns to predict a label, such as a cancer subtype, from the
features.

Feature selection chooses a subset of the features and gives the
classifier only those. It answers the question: which few features carry
the information about the label?

There are three reasons to do it.

1. **Better models.** Irrelevant features add noise. A model can fit that
   noise and then predict worse on new samples.
2. **Cheaper models.** Fewer features mean faster training and prediction.
   In medicine, they also mean fewer genes to measure for each new patient.
3. **Understanding.** A short list of genes is something a biologist can
   read and test in the lab.

## 1.2 Why gene expression data is hard

Gene expression measures how active each gene is in a tissue sample. One
measurement gives about 30,000 numbers per patient. A large clinical study
has a few thousand patients. In our data there are 2,976 patients and 28,552
usable genes: almost ten times more features than samples.

This setting is often written p ≫ n: many more features (p) than samples
(n). The same setting appears in text analysis and in image embeddings.
Five problems make it hard.

**Overfitting.** With more features than samples, a linear classifier can
separate any labeling of the training set perfectly, even random labels.
A perfect training score then says nothing.

**False discoveries.** Suppose no gene were related to the label. Testing
28,552 genes one by one at the usual 5% level would still flag about 1,400
genes by chance alone.

**Correlated features.** Genes work in groups. When one gene of a group
rises, hundreds of others rise with it. So many different gene sets can
predict equally well. So "the best genes" is not a single answer. The
project specification names this problem directly: removing all
correlated features may be impractical.

**Noise.** Both the biology and the measurement vary. The same tumour,
sequenced twice, does not give identical values (chapter 2).

**Cost.** Some methods train a classifier for every candidate feature, or
at every step. With 28,552 genes, such methods can take hours or days.

## 1.3 The goal of the project

The project specification (`references/MLDS_Feature_Selection.pdf`, Nov
2025) asks for four tasks:

1. Implement a feature selection pipeline for high-dimensional data.
2. Compare the results of different feature selection techniques.
3. Build infrastructure that speeds up the process, for example by
   running in parallel.
4. Run the pipeline on a second dataset, not necessarily gene expression.

The deliverables are software that runs on an input dataset, and a final
report. In this project, a configuration file drives the software.

## 1.4 What we built and asked

**The software.** `featsel` is a Python package of 16 selection methods.
Each one behaves exactly like a scikit-learn selector. A user with an
existing scikit-learn pipeline can swap their selector for ours and change
nothing else. A runner runs the whole comparison. One YAML file drives it.
It writes one record per experiment to a CSV file. See
[chapter 5](05_the_software.md).

**The main question.** For a given number of features `k`, which method
gives the best classifier? And by how much does it beat choosing `k`
features at random? The random control matters. If random genes do almost
as well, a method has not found much.

**The second question.** What does each method cost in time? Which
methods give the best quality for their cost? We answer with a
cost-quality frontier ([chapter 8](08_results_frontier.md)). It plots
selection time against AUC for every method on one plot.

**The targets.** The breast cancer data offers several labels: the PAM50
molecular subtype, estrogen receptor (ER) status, and survival. PAM50 is
the main target. Chapter 2 explains why. It also explains why PAM50
becomes five binary tasks.

## 1.5 Scope

This is a comparative study, not a new method. The project uses published
methods. It changes only one of them, HC2008: our version centres each
sorted p-value at its exact mean (section 4.4). The aim is a fair
comparison under one protocol, and a fast, reliable tool to run it.

## 1.6 What earlier work found

**Comparative studies on gene expression.** Dudoit et al. (2002) compared
classifiers on three tumour gene expression datasets. Simple methods, such
as diagonal linear discriminant analysis and nearest neighbours, did as
well as complex ones. Statnikov et al. (2005) compared classifiers and
gene selection methods on many multiclass cancer datasets. They found
multiclass SVMs the most accurate classifiers. Bolón-Canedo et al. (2014)
review the microarray datasets in common use and the selection methods
applied to them. Saeys et al. (2007) review selection methods for
bioinformatics as a whole.

**How our data differs.** Most of those datasets have tens to a few
hundred samples. SCAN-B has thousands of patients. So its test sets are
large enough to separate methods that score close together.

**Four known pitfalls.** The literature names four ways a comparison can
mislead. Chapter 3 shows how this project handles each.

| Pitfall | Source | Our answer |
|---|---|---|
| Selecting genes on all the data, then cross-validating | Ambroise and McLachlan (2002) | Selection inside every split (3.2) |
| No random control | Neekhra et al. (2025) | A random selector at every `k` (3.4) |
| Trusting the spread of k-fold scores | Bengio and Grandvalet (2004) | Repeated random splits (3.3) |
| Reporting scores without stability | Kuncheva (2007) | Kuncheva's index for every method (3.6) |

**Where this project stands.** It compares methods from all three
families (filters, embedded methods and wrappers), and Higher Criticism,
under one protocol. It runs on one large gene expression cohort and one
image dataset. This tests whether the ranking of methods carries over
between domains. A second image dataset, unseen during the design, then
tests the finished pipeline (section 8.8).

## 1.7 What this project adds

The project's contributions are evidence and tools, not a new method
(section 1.5).

1. **A comparison at a cohort size earlier studies lacked.** Most earlier
   comparisons had tens to hundreds of samples. Here 2,976 patients give
   test sets of about 1,500. Every method meets a random control at every
   `k`, and differences are tested with a paired test corrected for
   overlapping splits (chapter 7).
2. **Higher Criticism and Berk-Jones as selectors on real data.** Both
   thresholds were built for rare, independent signal. Our data has
   widespread, correlated signal. Higher Criticism stopped at the edge of
   its search on all three datasets. Berk-Jones stopped at or just before
   that edge on the two datasets it ran on. So both acted as top-10%
   filters (sections 7.5 and 8.8).
3. **A cost-quality frontier.** Selection time against AUC, for every
   method and setting on one plot. It shows which method to use for a time
   budget (chapter 8).
4. **Drop-in selectors that are tested, fast and exact.** All 16
   selectors are scikit-learn estimators. Those that copy a scikit-learn
   method pick the same features as it, on the CPU and on a GPU. Exact
   solvers make Lasso, RFE and Sequential selection faster. For Lasso and
   RFE they also reach the solution that scikit-learn's default solver
   stops short of (chapters 5 and 6).
5. **Checks of the data that change its use.** SCAN-B repeats some
   patients, and its labels are computed from the features. BreastMNIST
   holds copied images; we found them in its own images (chapter 2).

Section 9.3 repeats them in short.
