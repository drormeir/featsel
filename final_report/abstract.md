# Abstract

**Background.** Gene expression data describe each patient by tens of
thousands of genes, but a study has only hundreds or a few thousand
patients. A classifier trained on all the genes is slow and can learn
noise. Feature selection keeps a small set of features that still predicts
well. Many selection methods exist. It is not clear which method to use
for a given number of features, or how much better it is than random
features.

**Methods.** We built `featsel`, a Python package of 16 feature selection
methods: filters, embedded methods, wrappers, Higher Criticism, Berk-Jones
and a random control. Each method plugs into a scikit-learn pipeline. A
runner, driven by one configuration file, compares the methods under one
protocol. It uses 16 random splits with 70% of the samples in the training
set. It selects features inside each split, chooses the number of features
on validation sets, and measures ROC AUC on the test set. A random control
runs at every number of features. Paired tests are corrected for the
overlap between splits. We made the slow methods faster, with exact
solvers for RFE, RFECV and Lasso, and GPU paths for 14 selectors. The main
dataset is breast cancer gene expression (SCAN-B: 2,976 patients, 28,541
genes). It gives five tasks, each PAM50 subtype against the rest. The
second dataset is ResNet-50 embeddings of breast ultrasound images
(BreastMNIST: 780 images, 2,048 features). A third dataset, aircraft photos
(FGVC-Aircraft), was not seen during the design. It tested the finished
pipeline.

**Results.** Selection helps most when few features are allowed. At 10
features, the best methods beat random features by 0.10 to 0.20 AUC on
SCAN-B, and by 0.10 to 0.16 on the image data. As the number of features
grows, random features catch up, because thousands of correlated genes
carry the signal. At the number of features that each method chose, the
top methods cannot be told apart. On SCAN-B, eleven methods lie within
0.01 mean AUC. The Mann-Whitney U test, which scores one feature at a
time, is within 0.01 AUC of the best on all three datasets. It is among
the most stable methods, and on a GPU it takes less than a second per fit.
Selection rarely beats all the features. It never does on the image data.
On SCAN-B it does on one task of five, by 0.014 to 0.023 AUC. Higher
Criticism stopped at the edge of its search on all three datasets, because
the signal is not rare. The GPU filters ran 2.6 to 49 times faster than on
the CPU. The runner ran 3.2 to 5.3 times faster than plain scikit-learn.
Data checks found repeated patients in SCAN-B and copied images in
BreastMNIST. Both would leak between the training set and the test set.

**Conclusions.** On data like these, a simple test of one feature at a
time, run on a GPU, is almost as good as the complex methods, and it costs
far less. The best method for a given time budget cannot be known in
advance, but Mann-Whitney is a safe default. A selected gene list is not a
unique answer: many different sets predict equally well. Therefore,
stability must be reported next to accuracy. These results come from three
datasets, one classifier (logistic regression) and one laptop.
