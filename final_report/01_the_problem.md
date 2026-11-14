# 1. The problem

[Back to contents](README.md) · Next: [2. The data](02_the_data.md)

## 1.1 What feature selection is

A classifier learns to predict a label, such as a cancer subtype, from the
features of each sample. Here the samples are patients or images. The
features are genes, or numbers that describe an image.

Feature selection gives the classifier only the few features that carry
the information about the label. It has three uses:

1. **Better models.** Irrelevant features add noise that a model can fit.
   It then predicts worse on new samples.
2. **Cheaper models.** Training and prediction get faster. In medicine,
   fewer genes need to be measured for each new patient.
3. **Understanding.** A biologist can read a short gene list and test it
   in the lab.

## 1.2 Why gene expression data is hard

Gene expression measures how active each gene is in a tissue sample: about
30,000 numbers per patient. A large clinical study has a few thousand
patients. Our data has 2,976 patients and 28,541 usable genes, almost ten
times more features than samples. This setting, written p ≫ n, also
appears in text analysis and in image embeddings. Five problems make it
hard.

**Overfitting.** With more features than samples, a linear classifier can
perfectly separate any labeling of the training set, even random labels.
A perfect training score then says nothing.

**False discoveries.** If no gene were related to the label, testing
28,541 genes one by one at the usual 5% level would still flag about 1,400
by chance.

**Correlated features.** Genes work in groups: when one rises, hundreds
rise with it. Many different gene sets then predict equally well, so "the
best genes" has no single answer. The project specification notes that
removing all correlated features may be impractical.

**Noise.** Biology and measurement both vary. The same tumour, sequenced
twice, gives different values (chapter 2).

**Cost.** Some methods train a classifier for every candidate feature, or
at every step. With 28,541 genes they can take hours or days.

## 1.3 The goal of the project

The project specification (`references/MLDS_Feature_Selection.pdf`, Nov
2025) asks for four tasks:

1. Implement a feature selection pipeline for high-dimensional data.
2. Compare the results of different feature selection techniques.
3. Build infrastructure that speeds up the process, for example by
   running in parallel.
4. Run the pipeline on a second dataset, not necessarily gene expression.

The deliverables are software that runs on an input dataset, and a final
report. Here a configuration file drives the software.

## 1.4 What we built and asked

**The software.** `featsel` is a Python package of 16 selection methods.
Each one can replace a scikit-learn selector in an existing pipeline with
no other change. A runner, driven by one YAML file, runs the whole
comparison and writes one record per experiment to a CSV file
([chapter 5](05_the_software.md)).

**The main question.** For a given number of features `k`, which method
gives the best classifier, and by how much does it beat `k` random
features? If random genes do almost as well, a method has found little.

**The second question.** Which methods give the best quality for their
time cost? A cost-quality frontier answers this
([chapter 8](08_results_frontier.md)).

**The targets.** The breast cancer data offers several labels: the PAM50
molecular subtype, estrogen receptor (ER) status, and survival. PAM50 is
the main target, split into five binary tasks. Chapter 2 explains both
choices.

## 1.5 Scope

This is a comparative study, not a new method. It changes only one
published method, HC2008: our version centres each sorted p-value at its
exact mean (section 4.4). The aim is a fair comparison under one protocol,
and a fast, reliable tool to run it.

## 1.6 What earlier work found

**Comparative studies on gene expression.** Dudoit et al. (2002) compared
classifiers on three tumour gene expression datasets. Simple methods, such
as diagonal linear discriminant analysis and nearest neighbours, did as
well as complex ones. Statnikov et al. (2005) compared classifiers and
gene selection methods on many multiclass cancer datasets. Multiclass SVMs
were the most accurate classifiers. Bolón-Canedo et al. (2014) review
common microarray datasets and the selection methods applied to them.
Saeys et al. (2007) review selection methods for bioinformatics. Most of
those datasets have tens to a few hundred samples.

**Four known pitfalls.** Chapter 3 shows how this project avoids each.

| Pitfall | Source | Our answer |
|---|---|---|
| Selecting genes on all the data, then cross-validating | Ambroise and McLachlan (2002) | Selection inside every split (3.2) |
| No random control | Neekhra et al. (2025) | A random selector at every `k` (3.4) |
| Trusting the spread of k-fold scores | Bengio and Grandvalet (2004) | Repeated random splits (3.3) |
| Reporting scores without stability | Kuncheva (2007) | Kuncheva's index for every method (3.6) |

**Where this project stands.** It compares filters, embedded methods,
wrappers and Higher Criticism under one protocol. Running on a gene
expression cohort and an image dataset tests whether the ranking of
methods carries over between domains. A second image dataset, unseen
during the design, then tests the finished pipeline (section 8.8).

## 1.7 What this project adds

1. **A comparison at a cohort size earlier studies lacked.** Here 2,976
   patients give test sets of 893, large enough to separate
   methods that score close together. A paired test corrected for
   overlapping splits checks the differences (chapter 7).
2. **Higher Criticism and Berk-Jones as selectors on real data.** Both
   thresholds were built for rare, independent signal. On our widespread,
   correlated signal, both stopped at or near the edge of their search and
   acted as top-10% filters (sections 7.5 and 8.8).
3. **A cost-quality frontier.** One plot of selection time against AUC,
   for every method and setting. It shows which method fits a time budget
   (chapter 8).
4. **Drop-in selectors that are tested, fast and exact.** Those that copy
   a scikit-learn method pick the same features, on the CPU and on a GPU.
   Exact solvers make Lasso, RFE and Sequential selection faster. For
   Lasso and RFE they also reach the solution that scikit-learn's default
   solver stops short of (chapters 5 and 6).
5. **Checks of the data that change its use.** SCAN-B repeats some
   patients, and its labels are computed from the features. BreastMNIST
   holds copies of its own images (chapter 2).

Section 9.3 sums these up.
