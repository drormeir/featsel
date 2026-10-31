# Related Work

Draft. Full entries are in chapter 9, References.

## Three families of selection methods

Feature selection methods fall into three families (Guyon and Elisseeff
2003; Saeys et al. 2007). The families differ in how they use the
classifier.

**Filters** score each feature from the data alone, without a classifier.
They are fast and scale to tens of thousands of features. Most filters
score one feature at a time. So they ignore redundancy: two genes that carry
the same signal both rank high.

**Wrappers** search over feature subsets and score each subset by training
the classifier on it (Kohavi and John 1997). They see interactions between
features. They are the slowest family, and with few samples they can
overfit the search itself.

**Embedded methods** select features while the model is fitted. The lasso
penalizes the absolute size of the coefficients, which drives many of them to
exactly zero (Tibshirani 1996). A random forest ranks features by how much
they reduce impurity across its trees (Breiman 2001).

Saeys et al. (2007) review all three families for bioinformatics.

## The methods used here

**The Mann-Whitney U test** (Mann and Whitney 1947) compares two groups by
the ranks of their values. It does not assume a normal distribution. Its
statistic, divided by the product of the two group sizes, is the area under
the ROC curve of the feature used alone (Hanley and McNeil 1982). Ranking
genes by U-test p-value therefore ranks them by how well each one alone
separates the classes.

**Multiple testing.** A filter that tests every gene runs thousands of
tests. The Benjamini and Hochberg (1995) procedure keeps the expected share
of false discoveries below a chosen level. Bonferroni correction controls the
chance of even one false discovery and is stricter.

**SVM-RFE** (Guyon et al. 2002) is a widely used wrapper for gene
expression. It trains a linear SVM, removes the features with the smallest
weights, and repeats. It was introduced on gene expression data from
leukemia and colon cancer.

**Higher Criticism** comes from the detection of sparse signals. Donoho and
Jin (2004) define the HC statistic: it compares the sorted p-values of many
tests with the uniform distribution they would follow under the null. It
detects a small share of features with weak signal, the "rare and weak"
setting. Donoho and Jin (2008) turn the statistic into a threshold for
feature selection: keep the features whose p-values lie below the point
where HC is largest. The method picks its own number of features. Its
optimality results hold when the useful features are rare and weak and the
features are independent.

## Comparative studies on gene expression

Dudoit et al. (2002) compared classifiers on several tumour gene expression
datasets. Simple methods, such as diagonal linear discriminant analysis and
nearest neighbours, did as well as more complex ones. Statnikov et al.
(2005) compared classifiers and gene selection methods on many multiclass
cancer datasets and found multiclass SVMs to be the most accurate
classifiers. Bolón-Canedo et al. (2014) review the microarray datasets in
common use and the selection methods applied to them.

Most of these datasets have tens to a few hundred samples. SCAN-B has
thousands of patients, so its test sets are large enough to separate
methods that score close together.

## Pitfalls in evaluating selection methods

**Selection bias.** Ambroise and McLachlan (2002) showed that selecting
genes on the full dataset and then cross-validating the classifier gives
error rates near zero, even on data without signal. Selection must be
repeated inside every training split.

**The random control.** Neekhra et al. (2025) found that small random
feature subsets matched or beat feature selection on 28 of 30
high-dimensional datasets. A comparison without a random control cannot
tell whether a method finds signal or only reduces dimension.

**The spread of cross-validation scores.** Bengio and Grandvalet (2004)
showed that no unbiased estimator of the variance of k-fold
cross-validation exists. The fold scores are correlated because the
training sets overlap.

**Stability.** A method can score well but choose different features on
every split. Kuncheva (2007) proposes a consistency index between two
feature sets of equal size, corrected for the overlap expected by chance.

## Where this project stands

This project compares methods from all three families and Higher Criticism
under one protocol. It follows the four pitfalls above: selection inside
each split, a random control at every `k`, repeated random splits, and
Kuncheva's index for stability. It runs on one large gene expression
cohort and one image dataset, to test whether the ranking of methods
carries over between domains.
