# 9. Conclusions

[Back to contents](README.md) · Previous: [8. Results on image data](08_results_images.md) · Next: [10. Open work](10_open_work.md)

Three datasets and sixteen methods lead to one practical message. On data
like these, a simple test of one feature at a time, run on a GPU, is
almost as good as anything else, and it costs far less.

This chapter first sums up what was built and which decisions shaped the
results. It then asks what the datasets share: does one method lead at
every time budget, or can the best method for a budget not be known in
advance? It ends with the limits of the answer.

## 9.1 What we built

- **`featsel`**, a Python package of 16 feature selection methods covering
  filters, embedded methods, wrappers, Higher Criticism, Berk-Jones and a
  random control. Each is a scikit-learn estimator and drops into an existing
  pipeline. The ones that copy a scikit-learn method are tested to pick the
  same features. (Task 1.)
- **A config-driven runner** that compares methods under one protocol. It
  uses many random splits and selects inside each one. It runs a random
  control at every `k`. It tunes on validation sets drawn from the training
  set, so the test set reaches no choice. It writes one results table. It
  survives memory exhaustion and can resume a stopped run. (Tasks 1
  and 2.)
- **Speed**, in the runner and in the selectors (chapter 6). The runner
  is 4.3 to 10.6 times faster than plain scikit-learn on one core (6.8
  times against it on 12 threads), with identical scores. On a GPU, ten filters run 2.6 to 49 times faster and pick the
  same features. Running jobs in parallel gains at most 1.8 times on full
  SCAN-B on this laptop; shared memory bandwidth is the likely limit.
  (Task 3.)
- **A second dataset**, BreastMNIST image embeddings, run through the same
  pipeline unchanged. (Task 4.)
- **A third dataset, unseen during the design**: FGVC-Aircraft photo
  embeddings. It needed a loader and a config, but no change to the
  package. Every method ran on the GPU except tree importance, which has
  no GPU path. (Task 4.)

## 9.2 Key decisions

Ten decisions shaped the results. Each one is described in the section
given.

| Decision | Why | What it changed |
|---|---|---|
| Select features inside every split, never on all the data | Selection on all the data lets the test samples choose the features | On BreastMNIST with shuffled labels, selection on all the data raised the median AUC from 0.465 to 0.645 (section 3.2) |
| A random control at every `k` | A method has found something only if it beats random features of the same number | It showed that selection gains most at 10 features, and that random features catch up as `k` grows (sections 7.2, 8.2) |
| A t-test corrected for overlapping splits | Random splits share most of their training samples, so a plain paired t-test is overconfident | Established differences from all genes fell from 52 to 20 of 70 comparisons; 10 of the 20 are gains (sections 3.3, 7.3, 9.3 item 9) |
| The public GSE96058 data, one sample per patient, not the course copies | The course copies keep 93 repeated patients, and `scanb_small` lacks 11 of the 50 PAM50 genes | It removed a leak, and made the PAM50 recovery test possible (sections 2.1, 7.6) |
| The U test, not the F-test, for every filter | About half the gene values sit at a floor (in the first 200 genes), and only their order is meaningful | One valid test for all filters, so the comparison is between selection rules, not tests (section 4.3) |
| Find the copied images in BreastMNIST | A quarter of the images repeat another, and copies fall on both sides of a split | A leak removed: every BreastMNIST result in chapter 8 is without the copies (sections 2.2, 8.8) |
| Exact solvers, not liblinear, for RFE and RFECV, and a far tighter one (PPDNA) for Lasso | liblinear stops before the solution, and unless seeded it keeps different genes on each run | RFE became faster on data shaped like SCAN-B, and repeatable (section 6.2) |
| RFECV scored by AUC, not accuracy | Accuracy rewards the majority class on imbalanced tasks | RFECV rose from 0.720 to 0.962 mean AUC on SCAN-B (sections 7.1, 7.8) |
| Higher Criticism's search kept at 10% | Widening it to fit the data would tune the method to the answer | HC's behaviour on widespread signal is reported as it is (sections 4.4, 7.5) |
| A third dataset, unseen during the design | Two datasets shaped the design, so they cannot test it | The pipeline ran unchanged on FGVC-Aircraft (section 8.9) |

## 9.3 What we learned

The numbers come from the runs of chapters 7 and 8. The method
comparisons follow the protocol of chapter 3; the side studies keep the
settings stated in their sections.

**About the data.**

1. Read where the labels come from. PAM50 and the computed ER label are
   derived from the same gene expression we select from. A study that
   treats them as clinical facts overstates what it shows.
2. Check for repeated samples. 93 patients appear twice in SCAN-B's HiSeq
   data (section 2.1). In BreastMNIST, a quarter of the images repeat
   another; even MedMNIST's fixed split puts copies on both sides, in 57
   groups ([`copies_report.md`](../featsel_datasets/medmnist_breastmnist/copies_report.md)). The main results are without the copies (section 8.8); their effect
   is in [`demos/copies.ipynb`](../featsel_datasets/medmnist_breastmnist/demos/copies.ipynb).
3. The gene values are on an arbitrary scale. In the first 200 genes,
   about half the values are tied at a floor. The F-test's assumptions do not hold for such values. A rank
   test (Mann-Whitney U) does not need them, so it fits this data.

**About selection.**

4. **Selection matters most when few features are allowed.** At 10
   features, the filters, tree importance, the chains and Sequential beat
   random ones by 0.11 to 0.20 AUC on SCAN-B. On the image data, the
   filters and tree importance beat them by 0.10 to 0.16. The gain shrinks
   as `k` grows. On Basal it is gone by 250 genes, and on BreastMNIST by
   500 features. On LumA and LumB about 0.02 remains at 2,000 genes
   (sections 7.2, 8.2 and 8.9).
5. **The signal is widespread.** On SCAN-B, thousands of genes carry it,
   and they move together. On the image data, features are only weakly
   correlated, but a quarter or more of them pass the U test (sections
   2.2, 2.3). So random features catch up as `k` grows, and many
   different feature sets predict with no established difference in AUC.
6. **A gene list is therefore not a unique answer.** PAM50 is defined by
   50 genes. In their top 50, the methods found on average at most 12% of
   them (26% on the best task), yet they predicted the subtypes well
   (section 7.6). On SCAN-B, tree importance reached the same mean AUC as
   Mann-Whitney, 0.969, with no established difference on any task
   (p ≥ 0.07; section 7.1), but kept a different set on each split:
   stability 0.27 against 0.76, partly from its forest's own randomness
   (section 7.4). RFE has none, yet at other values of its C its genes
   had a Jaccard index of only 0.19 to 0.71 against C = 0.01, with no
   established change in AUC (section 7.9). So report stability next to accuracy (section 7.4).
7. **Simple is enough.** Mann-Whitney, one gene at a time, is within 0.002
   AUC of the best mean on SCAN-B, and within 0.012 of the best on both
   image datasets. It is among the most stable, and on a GPU it takes under
   a second per fit.
8. **The top methods are close.** On SCAN-B, eleven methods lie within
   0.01 mean AUC at their chosen `k`, and no method leads on every task.
   Of their 275 pairs on the five tasks, 51 give p < 0.05, 49 of them
   against Lasso, a chain or RFECV; among the top six, 2 of 75, no more
   than chance (section 7.1). On the image data, no pair of the methods
   within 0.01 of the best differs, and neither RFE nor Mann-Whitney has
   an established difference from all features (sections 8.1, 8.3, 8.9).
9. **Test with the splits' overlap in mind.** Random splits share most of
   their training patients, so their results are not independent. On
   SCAN-B, an uncorrected paired t-test passed 233 of 290 comparisons; the
   corrected resampled t-test passed 164. Against all genes, it cut the
   established differences from 52 to 20
   ([`scripts/correction_effect.py`](../scripts/correction_effect.py)).
10. **Higher Criticism needs rare signal.** On all three datasets HC2008
    stopped at the edge of its search, 10% of the features. Berk-Jones
    stopped at or just before it (sections 7.5, 8.5, 8.9). As a top-10% filter, both are among the best on SCAN-B and
    the most stable. HC2004 keeps one or two features and is the
    weakest method, except RFECV scored by accuracy on SCAN-B. On simulated rare signal (section 8.11), HC2008 kept
    a median of 33.5 features for 40 informative ones and had the best mean
    AUC, 0.858. Mann-Whitney with a validated `k` came within 0.015
    (p = 0.62).
11. **Wrappers pay only in some places, and only with care.** RFE had the
    highest mean AUC on both image datasets, with no established
    difference from all features or from the next methods (sections 8.1,
    8.9). At 250 features on FGVC-Aircraft it led the filters and tree
    importance (p = 0.003 to 0.047). On SCAN-B it trailed Mann-Whitney,
    mutual information and tree importance at every `k` (p ≤ 0.042;
    section 7.9), and was weak at 10 genes. A
    Mann-Whitney filter first raised RFE's mean AUC at 10 genes from 0.78
    to 0.95 (section 7.9). RFECV needs AUC as its inner score
    on imbalanced tasks (section 7.8). Sequential selection beat random but took 10 minutes
    per fit for 10 genes (section 7.10).

**About cost.**

12. **The cheap corner comes close to the best.** On SCAN-B, HC2008 and Berk-Jones reach
    0.970 mean AUC in 0.66 to 0.68 s per fit on the GPU. The best mean, mutual
    information at 0.971, takes 3.6 s; tree importance takes 17.5 s for
    0.969. On the image data every method except Sequential runs in about
    a second per fit or less (tree importance 1.1 s on FGVC-Aircraft,
    Appendix C.3), so
    cost does not decide the choice there.
13. **Choosing `k` costs more than selecting.** Choosing `k` on 5
    validation sets took Mann-Whitney 28 s per split on SCAN-B, against
    0.67 s for one fit. A method that chooses its own count skips this.
14. **A GPU helps sorting and matrix products, not everything.** A cheap
    GPU is slow in 64-bit arithmetic and has little memory. The data must
    also be copied to it. RFE stayed faster on the CPU. On this 4 GB GPU,
    the random forest did not fit at full size (section 6.2); a larger GPU
    is open work (section 10.1).
15. **Exactness has a price in agreement.** Faster, tighter solvers can pick
    different genes than scikit-learn's default. Lasso does on SCAN-B. RFE
    does on some synthetic data. The reason is that the default stops before it
    reaches the solution. So RFE, RFECV and Lasso run only their
    exact solvers (section 6.2).
16. **Selection buys a smaller model more often than a better one.** On
    BreastMNIST all 2,048 features scored 0.895 and on FGVC-Aircraft
    0.862: no selected set beat them. On SCAN-B, selection beat all 28,541
    genes by 0.014 to 0.023 on LumB, with ten methods, on all 16 splits.
    On the other four tasks no gain is established. So decide first
    whether you want a smaller model or a better one.

## 9.4 Across the datasets

Each dataset has its own frontier. What do they share?

- **What holds everywhere.** Selection beats random most at small `k`.
  Mann-Whitney on a GPU is within 0.012 AUC of the best, at the cheap end
  of the cost axis. HC2008 stops at the edge
  of its search. Wherever RFECV ran with both inner scores, scoring by
  AUC gave a higher mean than scoring by accuracy. The gain is
  established only on Her2, LumB and Normal of SCAN-B, where accuracy
  kept one gene (sections 7.8, 8.6).
- **What changes.** The best method for a given budget. At 250 features,
  RFE trails Mann-Whitney on SCAN-B by 0.020 AUC (p = 0.0003; section
  7.9) but leads it on FGVC-Aircraft by 0.029 (p = 0.047; section 8.9).
  Against tree importance the change is clearer: RFE trails by 0.026 on
  SCAN-B (p = 3 × 10⁻⁶) and leads by 0.036 on FGVC-Aircraft (p = 0.003),
  each on all 16 splits
  ([`figures/paired_tests_scanb70.csv`](figures/paired_tests_scanb70.csv),
  [`figures/paired_tests_aircraft70.csv`](figures/paired_tests_aircraft70.csv)).
  At 10
  features, the filters and tree importance, alone or chained, have the
  highest means on the image data, and Mann-Whitney then RFE on four
  SCAN-B tasks of five, where its lead is established over only some
  methods (section 7.9). Whether selection beats all features also changes:
  never on the image data, on one SCAN-B task of five.

So the best method for a budget can change from one dataset to another. What can be
known is a safe default: Mann-Whitney on a GPU is never far from the
best, and costs least.

## 9.5 Answer to the main question

For a given number of features `k`, which method gives the best
classifier, and by how much over random?

At small `k`, the answer depends on the dataset. At 10 genes on SCAN-B,
Mann-Whitney then RFE has the highest mean on four tasks of five, and the filters and
tree importance beat random genes by 0.12 to 0.19 AUC. At 10 features on the image data, the
filters and tree importance, alone or chained, lead, by 0.10 to 0.16 over
random.

At the `k` each method chose, the top methods are close. On SCAN-B,
eleven lie within 0.01 mean AUC, and among the top six no pair differs
more often than chance; on the image data, RFE and Mann-Whitney lie
within 0.012 of all features, with no established difference (section
9.3, item 8).

If time also counts, a GPU filter is the practical choice. On SCAN-B,
HC2008 or Mann-Whitney reaches 0.970 or 0.969 mean AUC in under a second
per fit, 0.001 to 0.002 below the best.

## 9.6 Limits

These conclusions hold under the conditions below. Each limit is
described in the section given.

**The data.**

- Three datasets. Two of them are image embeddings from the same network,
  ResNet-50 (chapter 2). The results may not hold on other data.
- Every SCAN-B task predicts PAM50, a label computed from the genes, not
  measured (section 2.1). No measured target was run (section 10.1).
- The project's rule is to use only data without duplicates. The BreastMNIST results
  of chapter 8 are without the copied images (section 8.8). Sequential
  selection has not yet run without them. The side studies on the image
  data (RFE's C, the classifier's C, the tree count) still use the copies
  (section 10.1). So do the choice of the V2 weights (without the copies, V2 is
  not established as better than V1, p = 0.36; section 2.2), the
  shuffled-label demo (section 3.2) and the BreastMNIST data facts
  (section 2.2).
- FGVC-Aircraft holds one group of three near-identical photos, one
  aircraft shot seconds apart, with the same label (section 2.3). All its
  runs include them.

**The protocol.**

- One classifier, logistic regression, with C fixed at 1 (section 3.8).
  Other values of C were tested only for Mann-Whitney and all features,
  only on the image data. The classifier study used other settings
  (section 7.7).
- One training fraction, 70%, and 16 splits. The spread over splits is not
  a confidence interval for new patients (section 3.3).
- The five SCAN-B tasks share patients, so they are not independent
  replicates.
- Each selector's own settings are fixed, except Lasso's C (section 3.11).
  HC keeps alpha0 = 0.1 (section 4.4). The correlation results come
  from a previous redundancy rule; at each `k`, the current rule changes
  at most one gene of a selection; the chosen `k` and the AUC were not
  recomputed (section 4.2).
- A chain's first count is fixed: 1,000 genes on SCAN-B, 500 features on
  the image data. Other counts were tested only on the image data
  (section 8.7).
- Tree importance uses 200 trees. Its ranking depends on the forest's
  seed; whether more trees change its AUC was not tested (section 7.4).
- The decision threshold is fixed at 0.5 on the predicted probability.
  Macro-F1 and balanced accuracy depend on it; AUC does not.
- Each test is judged at p < 0.05, with no correction for the number of
  tests, more than a hundred per dataset (section 3.3). Some established
  differences may be chance, most likely those with p between 0.01 and
  0.05, such as RFE ahead of Mann-Whitney on FGVC-Aircraft (p = 0.047).
  Where a count of tests matters, section 7.1 also gives it with a
  Bonferroni correction.
- The runner stores each cell's metrics, not each test patient's score, so
  a new metric needs a new run (section 10.1).
- Side studies keep their own protocols: the PAM50 recovery uses 10
  splits (section 7.6), the classifier study 12 splits at 50% (section
  7.7).

**The cost.**

- Every time comes from one laptop with a 4 GB consumer GPU (chapter 6).
  A time is meaningful only on that machine.
- The runner's speed-up over scikit-learn comes from three runs of one
  small grid, with one fit per selection, not the 10 of the experiments
  (section 6.1).

**Higher Criticism.** No real dataset here has rare signal. One
simulation tests HC in the setting it was built for (section 8.11): one
setting, with independent normal features.
