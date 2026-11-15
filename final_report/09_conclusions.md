# 9. Conclusions

[Back to contents](README.md) · Previous: [8. Results on image data](08_results_images.md) · Next: [10. Open work](10_open_work.md)

Each dataset is its own classification problem, with its own frontier of
cost against AUC (chapters 7 and 8). This chapter asks what the frontiers
share: does one method lead at every time budget on every dataset, or can
the best method for a budget not be known in advance?

## 9.1 What we built

- **`featsel`**, a Python package of 16 feature selection methods covering
  filters, embedded methods, wrappers, Higher Criticism, Berk-Jones and a
  random control. Each is a scikit-learn estimator and drops into an existing
  pipeline. The ones that copy a scikit-learn method are tested to pick the
  same features. (Task 1.)
- **A config-driven runner** that compares methods under one protocol. It
  uses many random splits and selects inside each one. It runs a random
  control at every `k`. It tunes on validation sets drawn from the training
  set, so the test set is used once. It writes one results table. It
  survives memory exhaustion and can resume a stopped run. (Tasks 1
  and 2.)
- **Speed**, in the runner and in the selectors. The runner is 3.2 to
  5.3 times faster than plain scikit-learn, with identical scores. Ten
  selectors run on a GPU and pick the same features as on the CPU. The
  filters run 2.6 to 49 times faster there. Exact solvers make Lasso, RFE,
  RFECV and Sequential selection faster. Running jobs in parallel gains at
  most 1.8 times on full SCAN-B on this laptop. From 3 workers on, the jobs
  slow each other down. This points to shared memory bandwidth
  (section 6.1). (Task 3.)
- **A second dataset**, BreastMNIST image embeddings, run through the same
  pipeline unchanged. (Task 4.)
- **A third dataset, unseen during the design**: FGVC-Aircraft photo
  embeddings. It needed a loader and a config, but no change to the
  package. It ran entirely on the GPU in 17 minutes. (Task 4.)

## 9.2 What we learned

The numbers come from the runs of chapters 7 and 8, all under the protocol
of chapter 3.

**About the data.**

1. Read where the labels come from. PAM50 and the computed ER label are
   derived from the same gene expression we select from. A study that
   treats them as clinical facts overstates what it shows.
2. Check for repeated samples. 93 patients appear twice in SCAN-B's HiSeq
   data. Random splits put them on both sides and leak. We kept one
   sample per patient. In BreastMNIST, a quarter of the images repeat
   another, as a 2023 letter reports. Even MedMNIST's fixed split puts
   copies on both sides, in 57 groups. We found the copies in
   BreastMNIST's own images and reran without them. All features and the
   best methods lost about 0.03 AUC. The order of methods changed little
   (section 8.8).
3. The gene values are on an arbitrary scale, with half the values tied at
   a floor. The F-test's assumptions do not hold for such values. A rank
   test (Mann-Whitney U) does not need them, so it fits this data.

**About selection.**

4. **Selection matters most when few features are allowed.** At 10
   features, the good methods beat random ones by 0.10 to 0.20 AUC on
   SCAN-B, and by 0.10 to 0.16 on the image data. The gain shrinks as `k`
   grows. On Basal and on BreastMNIST it is gone by 250 to 1,000. On LumA
   and LumB it lasts to 2,000 genes (sections 7.2 and 8.2).
5. **The signal is widespread.** Thousands of genes carry it, and they
   move together. So random genes catch up as `k` grows, and many
   different gene sets predict equally well.
6. **A gene list is therefore not a unique answer.** On SCAN-B, tree
   importance scored as well as Mann-Whitney (0.969 mean AUC) but kept a
   different set on each split: stability 0.27 against 0.76. So report
   stability next to accuracy (section 7.4).
7. **Simple is enough.** Mann-Whitney, one gene at a time, is within 0.002
   AUC of the best mean on SCAN-B, and within 0.01 of the best on both
   image datasets. It is among the most stable, and on a GPU it takes under
   a second per fit.
8. **The top methods cannot be told apart.** On SCAN-B, eleven methods lie
   within 0.01 mean AUC at their chosen `k`, and no method leads on every
   task. On the image data, RFE ties all features and Mann-Whitney is
   within 0.01.
9. **Test with the splits' overlap in mind.** Random splits share most of
   their training patients, so their results are not independent. On
   SCAN-B, an uncorrected paired t-test passed 233 of 290 comparisons; the
   corrected resampled t-test passed 164. Against all genes, it cut the
   established differences from 52 to 20.
10. **Higher Criticism needs rare signal.** On all three datasets HC2008
    stopped at the edge of its search, 10% of the features. Berk-Jones
    stopped at or just before it. Their signal is not rare. As a top-10%
    filter, both are among the best on SCAN-B (0.970) and the most stable
    (0.92). HC2004 keeps one or two features and is the weakest method.
11. **Wrappers pay only in some places, and only with care.** RFE was the
    best method on both image datasets at large `k`, tied with all
    features. On SCAN-B it was below the filters, and at 10 genes it scored
    below random genes on two tasks (not established). A Mann-Whitney filter first raised RFE's
    mean AUC at 10 genes from 0.78 to 0.95, and made it the best method at
    10 genes on four SCAN-B tasks. RFECV's default inner score, accuracy,
    made it keep one gene on three imbalanced tasks. Scored by AUC it
    recovered (0.962 on SCAN-B). Sequential selection beat random but took
    10 minutes per fit for 10 genes.

**About cost.**

12. **The cheap corner wins.** On SCAN-B, HC2008 and Berk-Jones reach
    0.970 mean AUC in 0.66 s per fit on the GPU. The best mean, mutual
    information at 0.971, takes 3.6 s; tree importance takes 16.6 s for
    0.969. On the image data every method except Sequential runs in under
    a second, so cost does not decide the choice there.
13. **Choosing `k` costs more than selecting.** Choosing `k` on 5
    validation sets took Mann-Whitney 26 s per split on SCAN-B, against
    0.67 s for one fit. A method that chooses its own count skips this.
14. **A GPU helps sorting and matrix products, not everything.** A cheap
    GPU is slow in 64-bit arithmetic and has little memory. The data must
    also be copied to it. RFE stayed faster on the CPU. The random forest
    could not move to the GPU.
15. **Exactness has a price in agreement.** Faster exact solvers can pick
    different genes than scikit-learn's default. Lasso does on SCAN-B. RFE
    does on some synthetic data, and in 14 of 100 cells of a preliminary
    SCAN-B frontier run. The reason is that the default stops before it
    reaches the solution. AUC did not change. So RFE, RFECV and Lasso now
    run only their exact solvers (section 6.2).
16. **Selection buys a smaller model more often than a better one.** On
    BreastMNIST all 2,048 features scored 0.924 and on FGVC-Aircraft
    0.862: no selected set beat them. On SCAN-B, selection beat all 28,541
    genes by 0.014 to 0.023 on LumB, with ten methods, on all 16 splits.
    On the other four tasks no gain is established. So decide first
    whether you want a smaller model or a better one.

## 9.3 Across the datasets

Each dataset has its own frontier. What do they share?

- **What holds everywhere.** Selection beats random most at small `k`.
  Mann-Whitney on a GPU is within 0.01 AUC of the best, at the cheap end
  of the cost axis. HC2008 stops at the edge
  of its search. RFECV needs AUC as its inner score.
- **What changes.** The best method for a given budget. At large `k`, RFE
  leads on both image datasets but trails the filters on SCAN-B. At 10
  features, the filters lead on the image data, and Mann-Whitney then RFE
  leads on SCAN-B. Whether selection beats all features also changes:
  never on the image data, on one SCAN-B task of five.

So the best method for a budget cannot be known in advance from these
three datasets. What can be known is a safe default: Mann-Whitney on a GPU
is never far from the best, and costs least.

## 9.4 What this project adds

Section 1.7 lists the contributions. In short:

- A method comparison on a cohort of 2,976 patients, against a random
  control at every `k`, with paired tests corrected for overlapping
  splits. The correction removed most of the claimed gains over all genes.
- Evidence that Higher Criticism and Berk-Jones, used as selectors on
  data with widespread signal, reduce to a fixed top-10% filter.
- A cost-quality frontier per dataset, over every method and setting.
- A package of 16 drop-in selectors, tested against scikit-learn, with
  GPU paths and exact solvers.
- Data checks that changed how the data was used: repeated patients,
  computed labels, and copied images found in BreastMNIST itself.

## 9.5 Answer to the main question

For a given number of features `k`, which method gives the best
classifier, and by how much over random?

At small `k`, the answer depends on the dataset. At 10 genes on SCAN-B,
Mann-Whitney then RFE is best on four tasks of five, and the filters beat
random genes by 0.12 to 0.19 AUC. At 10 features on the image data, the
filters and tree importance lead, by 0.10 to 0.16 over random.

At the `k` each method chose, the top methods cannot be told apart. On
SCAN-B, eleven lie within 0.01 mean AUC; on the image data, RFE and
Mann-Whitney lie within 0.01 of all features.

If time also counts, a GPU filter is the practical choice. On SCAN-B,
HC2008 or Mann-Whitney reaches 0.970 or 0.969 mean AUC in under a second
per fit, 0.001 to 0.002 below the best.
