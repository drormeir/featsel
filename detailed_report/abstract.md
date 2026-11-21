# Abstract

**Background.** Gene expression data describe each patient by tens of
thousands of genes, but a study has only hundreds or a few thousand
patients. A classifier trained on all the genes is slow and can learn
noise. Feature selection keeps a small set of features that still predicts
well. Many selection methods exist. It is not clear which method to use
for a given number of features, or how much better it is than random
features.

**Methods.** We built `featsel`, a Python package of 16 feature selection
methods that plug into scikit-learn pipelines. A runner compares the
methods under one protocol: 16 random splits, selection inside each split,
the number of features chosen on validation sets, and ROC AUC (area under
the ROC curve: 0.5 is chance, 1 is perfect) on the test set. A random control runs at every number of features. Paired tests are
corrected for the overlap between splits. We made the slow methods faster,
with exact solvers and GPU paths. The main dataset is breast cancer gene
expression (SCAN-B: 2,976 patients, 28,541 genes, five PAM50 subtype
tasks). The second is ResNet-50 embeddings of breast ultrasound images
(BreastMNIST: 585 images without copies, 2,048 features). A third, aircraft photos
(FGVC-Aircraft), was not seen during the design. It tested the finished
pipeline.

**Results.** Selection helps most when few features are allowed. At 10
features, the best methods beat random features by 0.11 to 0.20 AUC on
SCAN-B, and by 0.10 to 0.16 on the image data. As the number of features
grows, random features catch up, because thousands of correlated genes
carry the signal. At the number of features that each method chose, the
top methods cannot be told apart: on SCAN-B, eleven lie within 0.01 mean
AUC. The Mann-Whitney U test, which scores one feature at a time, is
within 0.012 AUC of the best on all three datasets, and on a GPU it takes
less than a second per fit. Selection rarely beats all the features: never
on the image data, and on one SCAN-B task of five. The GPU filters ran 2.6
to 49 times faster than the originals they replace. Data checks found repeated patients in
SCAN-B and copied images in BreastMNIST, which would leak between the
training set and the test set.

**Conclusions.** On data like these, a simple test of one feature at a
time, run on a GPU, is almost as good as the complex methods, and it costs
far less. The best method for a given time budget cannot be known in
advance, but Mann-Whitney is a safe default. Many different gene sets
predict with no established difference in AUC, so stability must be
reported next to accuracy.
These results come from three datasets, one classifier and one laptop.
