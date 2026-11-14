# Appendix A. Glossary

[Back to contents](README.md)

**AUC (ROC AUC).** The area under the ROC curve: the chance that a model
scores a random positive sample above a random negative one. 0.5 is
guessing, 1.0 is perfect. It needs no decision threshold.

**Balanced accuracy.** The mean recall (true-positive rate) over the
classes. A model that predicts one class for every sample scores 0.5.

**Benjamini-Hochberg.** A procedure that keeps the expected share of
false discoveries among the selected features below a chosen level (FDR).

**Berk-Jones threshold.** A relative of Higher Criticism. It scores the
i-th smallest p-value by the null chance of an i-th smallest p-value that
small. It keeps the features up to the rank where that chance is smallest
(section 4.5).

**Bonferroni correction.** Testing m hypotheses at level α / m each, so
the chance of even one false positive stays below α.

**Bootstrap sample.** A sample of the same size drawn with replacement.
Each tree of a random forest grows on one.

**Cell.** One experiment: one task, split, method, `k` and classifier.
It is one line of the runner's results CSV.

**Control (random selection).** A method that picks `k` features at
random. A real method is useful only if it beats it.

**Corrected resampled t-test.** A paired t-test over random splits that
allows for their shared training samples (Nadeau and Bengio 2003). It
enlarges the variance of the mean difference by a factor that grows with
the ratio of test set size to training set size.

**Cost-quality frontier.** The methods that no other method beats on both
time and quality at once.

**Embedding.** A vector that a neural network produces to describe an
input, such as an image. Here, the 2,048 numbers ResNet-50 gives before
its last layer.

**Embedded method.** A method that selects while fitting a model, such as
Lasso or a random forest.

**F-test (ANOVA).** A test of whether a feature's class means differ,
assuming normal values with equal spread.

**Feature.** One measured quantity per sample: a gene, or one embedding
number.

**Filter.** A method that scores features from the data alone, without a
classifier.

**FPKM.** Fragments per kilobase of transcript per million mapped reads:
a normalized count of how much of a gene's RNA was read.

**GPU.** Graphics processing unit: a processor with hundreds or thousands
of small cores. It is fast at one operation on many numbers at once.

**Higher Criticism (HC).** A method that compares sorted p-values with
what pure noise would give. It keeps the features up to the point of
largest difference, so it picks its own number of features.

**k.** The number of features a method is asked to keep.

**Kuncheva index.** How similar the feature sets chosen on different
splits are, corrected for chance: 1 = identical, 0 = chance.

**Lasso.** A linear model with an L1 penalty, λ·Σ|wᵢ|, which sets many
weights to exactly zero. Here the model is logistic regression.
scikit-learn's C is 1/λ, so a smaller C means a stronger penalty. In the
frontier, Lasso chooses C by AUC on validation sets.

**Leakage.** Information from the test set reaching training or
selection. It makes test scores look better than they are.

**Macro-F1.** The F1 score of each class, averaged with equal weight. F1
is the harmonic mean of precision and recall.

**Mann-Whitney U test.** A rank test of whether one group's values tend
to be larger than the other's. For two classes, U / (n₁n₂) is the
feature's AUC.

**Median and quartiles.** The middle value, and the values a quarter and
three quarters of the way up. One unusual split barely moves them.

**Monte Carlo split.** A random division of the samples into a training
set and a test set, drawn afresh each time.

**Mutual information.** How much knowing a feature reduces uncertainty
about the label. It catches any dependence, not only a linear one.

**Newton's method.** An optimization method that uses second derivatives
to take large, accurate steps, so it converges in few iterations. RFE and
RFECV solve their linear SVM with it.

**Nogueira's stability.** A stability measure built from how often each
feature is selected across splits (Nogueira et al. 2018). Unlike the
Kuncheva index, it allows sets of different sizes. 1 = identical sets,
0 = chance.

**p ≫ n.** Many more features (p) than samples (n).

**PAM50.** A published classifier that assigns a breast tumour to one of
five subtypes from 50 genes. Here each subtype against the rest is a
separate binary task.

**Paired test.** A test on the differences between two methods measured
on the same splits. It removes the noise both share.

**p-value.** The chance, if there were no real effect, of a result at
least this extreme.

**Random forest.** Many decision trees that vote together. Each tree
grows on a bootstrap sample and tries a random subset of the features at
each split.

**Rank once.** Fitting a method once per split and taking every `k` from
its single ranking, as mutual information and correlation do.

**RFE (recursive feature elimination).** A wrapper that trains a model,
removes the features with the smallest weights, and repeats.

**RFECV.** RFE that picks the number of features by inner
cross-validation.

**Sample.** One unit the data describes: a patient or an image.

**scikit-learn.** The standard Python machine learning library. Our
selectors follow its interface.

**Sequential forward selection.** A wrapper that adds the best single
feature at each step, judged by cross-validation.

**Spearman correlation.** The correlation of the ranks of two variables.
No increasing transform changes it.

**Stability.** Whether a method chooses the same features when the
training data changes a little.

**Stratified.** A split that keeps the whole data's class proportions in
both the training set and the test set.

**SVM (support vector machine).** A classifier that finds the boundary
with the widest margin between classes. In RFE, a linear SVM's weights
rank the features.

**Task.** One binary prediction problem. SCAN-B gives five separate
tasks, each subtype against the rest. BreastMNIST and FGVC-Aircraft give
one each.

**Test set.** The samples of a split kept apart from all fitting,
selection and tuning. It is used once, to score the final model.

**Training set.** The samples of a split used to select features and fit
the classifier.

**Validation set.** A part of the training set held back to choose a
setting, such as `k` or Lasso's C. For these two, the training set is cut
into 5 folds, and each fold serves once as the validation set.

**Wilcoxon signed-rank test.** A paired test that ranks the absolute
differences and checks whether positive and negative ones balance. It
assumes independent pairs, which random splits are not.

**Wrapper.** A method that searches over feature subsets by training a
classifier on each.
