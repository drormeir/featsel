# 1. The problem

[Back to contents](README.md) · Next: [2. The data](02_the_data.md)

## 1.1 What feature selection is

A dataset is a table. Each row is a sample, here a patient or an image.
Each column is a feature, here a gene or one number that describes an
image. A classifier learns to predict a label, such as a cancer subtype,
from the features.

Feature selection chooses a subset of the features and gives the
classifier only those. It answers the question: which few columns carry
the information about the label?

There are three reasons to do it.

1. **Better models.** Irrelevant features add noise. A model can fit that
   noise and then predict worse on new samples.
2. **Cheaper models.** Fewer features mean faster training and prediction,
   and, in medicine, fewer genes to measure for each new patient.
3. **Understanding.** A short list of genes is something a biologist can
   read and test in the lab.

## 1.2 Why gene expression data is hard

Gene expression measures how active each gene is in a tissue sample. One
measurement gives about 30,000 numbers per patient. A large clinical study
has a few thousand patients. In our data there are 2,976 patients and 28,552
usable genes: almost ten times more features than samples.

This setting is often written p ≫ n: many more features (p) than samples
(n). It causes four problems.

**Overfitting.** With more features than samples, a linear classifier can
separate any labeling of the training samples perfectly, even random
labels. A perfect training score then says nothing.

**False discoveries.** Testing 28,552 genes one by one at the usual 5%
level would flag about 1,400 genes by chance alone, even if no gene were
related to the label.

**Correlated features.** Genes work in groups. When one gene of a group
rises, hundreds of others rise with it. So many different gene sets can
predict equally well, and "the best genes" is not a single answer. The
project specification names this problem directly: removing all
correlated features may be impractical.

**Cost.** Some methods train a classifier for every candidate feature or
every step. With 28,552 genes, such methods can take hours or days.

## 1.3 The goal of the project

The project specification (`references/MLDS_Feature_Selection.pdf`, Nov
2025) asks for four tasks:

1. Implement a feature selection pipeline for high-dimensional data.
2. Compare the results of different feature selection techniques.
3. Build infrastructure that speeds up the process, for example by
   running in parallel.
4. Run the pipeline on a second dataset, not necessarily gene expression.

The deliverables are software driven by a configuration file, and this
report.

## 1.4 What we built and asked

**The software.** `featsel` is a Python package of 15 selection methods.
Each one behaves exactly like a scikit-learn selector. A user with an
existing scikit-learn pipeline can swap their selector for ours and change
nothing else. A runner, driven by one YAML file, runs the whole comparison
and writes one row per experiment to a CSV table. See
[chapter 5](05_the_software.md).

**The main question.** For a given number of features `k`, which method
gives the best classifier, and by how much does it beat choosing `k`
features at random? The random control matters: if random genes do almost
as well, a method has not found much.

**The second question.** What does each method cost in time, and which
methods give the best quality for their cost? We answer it with a
cost-quality frontier ([chapter 8](08_results_frontier.md)): selection time
against AUC, every method on one plot.

## 1.5 Scope

This is a comparative study, not a new method. The project uses published
methods only. The aim is a fair comparison under one protocol, and a fast,
reliable tool to run it. `SCOPE.md` in the repository records what is in
and out of scope.
