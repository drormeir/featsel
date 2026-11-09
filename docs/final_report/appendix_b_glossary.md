# Appendix B. Glossary

[Back to contents](README.md)

**AUC (ROC AUC).** The area under the ROC curve. Equal to the chance that
a model scores a random positive sample above a random negative one. 0.5
is guessing, 1.0 is perfect. It needs no decision threshold.

**Balanced accuracy.** The average of the recall (true-positive rate) of
each class. Always 0.5 for a model that predicts one class for everyone.

**Benjamini-Hochberg.** A procedure that keeps the expected share of false
discoveries among the selected features below a chosen level (FDR).

**Bonferroni correction.** Testing m hypotheses at level α / m each, so
the chance of even one false positive stays below α.

**Bootstrap sample.** A sample of the same size drawn with replacement.
Each tree of a random forest grows on one.

**Cell.** One experiment: one task, split, method, `k` and classifier.
One row of the runner's CSV.

**Control (random selection).** A method that picks `k` features at
random. Every real method must beat it.

**Cost-quality frontier.** The methods that no other method beats on both
time and quality at once.

**Embedding.** A vector of numbers that a neural network produces to
describe an input, such as an image. Here, the 2,048 numbers ResNet-50
gives before its last layer.

**Embedded method.** A selection method that selects as part of fitting a
model, such as Lasso or a random forest.

**F-test (ANOVA).** A test of whether the class means of a feature differ,
assuming normal values with equal spread.

**Feature.** One column of the data: a gene, or one embedding number.

**Filter.** A selection method that scores features from the data alone,
without a classifier.

**FPKM.** Fragments per kilobase of transcript per million mapped reads: a
normalized count of how much of a gene's RNA was read.

**GPU.** Graphics processing unit: a processor with thousands of small
cores, fast at doing the same operation on many numbers at once.

**Higher Criticism (HC).** A method that compares sorted p-values with
what pure noise would give, and keeps the features up to the point of
largest difference. It picks its own count.

**k.** The number of features a method is asked to keep.

**Kuncheva index.** A measure of how similar the feature sets chosen on
different splits are, corrected for chance: 1 = identical, 0 = chance.

**Lasso.** Regression with an L1 penalty, C·Σ|wᵢ|, which sets many weights
to exactly zero.

**Leakage.** Information from the test samples reaching the training,
which makes test scores look better than they are.

**Macro-F1.** The F1 score of each class, averaged with equal weight. F1
is the harmonic mean of precision and recall.

**Mann-Whitney U test.** A rank test of whether one group's values tend to
be larger than the other's. For two classes, U / (n₁n₂) is the feature's
AUC.

**Median and quartiles.** The middle value, and the values a quarter and
three quarters of the way up. Robust to an odd split.

**Monte Carlo split.** A random division into training and test parts,
drawn afresh each time.

**Mutual information.** How much knowing a feature reduces uncertainty
about the label. Catches any kind of dependence, not only linear.

**Newton's method.** An optimization method that uses second derivatives
to take large, accurate steps. Converges in few iterations.

**p ≫ n.** Many more features (p) than samples (n).

**PAM50.** A published classifier that assigns a breast tumour to one of
five subtypes from 50 genes.

**Paired test.** A test on the differences between two methods measured on
the same splits. It removes the noise shared by both.

**p-value.** The chance, if there were no real effect, of a result at
least this extreme.

**Random forest.** Many decision trees, each on a bootstrap sample and a
random subset of features at each split, voting together.

**Rank once.** Fitting a method once per split and taking every `k` from
its single ranking.

**RFE (recursive feature elimination).** A wrapper that trains a model,
removes the features with the smallest weights, and repeats.

**RFECV.** RFE that picks the number of features by inner
cross-validation.

**Sample.** One row of the data: a patient or an image.

**scikit-learn.** The standard Python machine learning library. Our
selectors follow its interface.

**Sequential forward selection.** A wrapper that adds the single best
feature at each step, judged by cross-validation.

**Spearman correlation.** The correlation of the ranks of two variables.
Unchanged by any increasing transform.

**Stability.** Whether a method chooses the same features when the
training data changes a little.

**Stratified.** A split that keeps the class proportions of the whole data
in each part.

**SVM (support vector machine).** A classifier that finds the boundary
with the widest margin between classes. A linear SVM's weights rank
features in RFE.

**Task.** One binary prediction problem. SCAN-B gives five (each subtype
against the rest); BreastMNIST gives one.

**Wilcoxon signed-rank test.** A paired test that ranks the absolute
differences and checks whether positive and negative ones balance.

**Wrapper.** A selection method that searches over feature subsets by
training a classifier on each.
