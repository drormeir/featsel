# 9. Conclusions

[Back to contents](README.md) · Previous: [8. Results: cost against quality](08_results_frontier.md) · Next: [10. Open work](10_open_work.md)

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
  package. It ran entirely on the GPU in 5 minutes. (Task 4.)

## 9.2 What we learned

The numbers below come from the runs of chapters 7 and 8, made before
RFE, RFECV and Lasso dropped liblinear and before `k` was chosen on
validation sets. They will be updated after the rerun.

**About the data.**

1. Read where the labels come from. PAM50 and the computed ER label are
   derived from the same gene expression we select from. A study that
   treats them as clinical facts overstates what it shows.
2. Check for repeated samples. 93 patients appear twice in SCAN-B's HiSeq
   data. Random splits put them on both sides and leak. We kept one
   sample per patient. In BreastMNIST, a quarter of the images repeat
   another, as a 2023 letter reports. Even MedMNIST's fixed split puts
   copies on both sides, in 57 groups. We found the copies in
   BreastMNIST's own images and reran without them. AUC fell by 0.01 to
   0.03, and the fine ranking of methods changed. The main conclusions
   held.
3. The gene values are on an arbitrary scale, with half the values tied at
   a floor. The F-test's assumptions do not hold for such values. A rank
   test (Mann-Whitney U) does not need them, so it fits this data.

**About selection.**

4. **Selection matters most when few features are allowed.** At k = 5 the
   best methods beat random genes by 0.26 to 0.29 macro-F1. By k = 250 the
   gain is a few hundredths, and some methods fall below random at
   k = 1000.
5. **The signal is widespread.** Thousands of genes carry it, and they
   move together. So random genes catch up fast as `k` grows. Many
   different gene sets predict equally well.
6. **A gene list is therefore not a unique answer.** PAM50 is defined by
   50 genes. In their top 50, the methods found on average at most 12% of
   them (26% on the best task). Yet they predicted the subtypes well. Tree
   importance had the highest median score and was the least stable. So
   report stability next to accuracy.
7. **Simple is enough.** A one-gene-at-a-time rank test is within 0.03
   macro-F1 of the best method on SCAN-B at every `k`. It is by far the
   most stable. On a GPU it takes under a second.
8. **The top methods cannot be told apart.** Tree importance had the
   highest median on gene expression. Lasso and tree importance shared it
   on image embeddings. A paired test corrected for overlapping splits
   establishes neither lead. On the frontier's 16 splits, scored by AUC,
   it establishes two small leads. Tree importance beats Mann-Whitney at
   k = 250. It also beats Lasso at k = 1000. Both leads are under 0.01 AUC
   (section 8.3). The broad pattern held on both datasets. Selection beats
   random most at small `k`, and RFE is weakest at small `k`.
9. **Test with the splits' overlap in mind.** Random splits share most of
   their training patients, so their results are not independent. An
   uncorrected test passed 67 of 81 comparisons on SCAN-B. The corrected
   resampled t-test passed 35.
10. **Higher Criticism needs rare signal.** On all three datasets it
    stopped at the edge of its search range. So it acted as a fixed
    top-10% filter. It assumes rare and independent signal, and that does
    not hold here. Berk-Jones judges each sorted p-value by its exact null
    distribution. It kept a subset of the same genes and scored the same.
11. **Wrappers did not pay.** RFE, RFECV and Sequential selection were
    among the slowest methods. They scored below the simple filters on
    SCAN-B. RFECV's default inner score is accuracy. With it, RFECV kept
    one or two genes on imbalanced tasks. With AUC as its inner score, it
    recovered (0.961 on SCAN-B). A wrapper is only as good as its inner
    score. A Mann-Whitney filter before RFE raised RFE's AUC at 10 genes
    from 0.81 to 0.95 on SCAN-B, in half the time. On BreastMNIST, the
    same filter lowered the AUC of Lasso and tree importance.

**About cost.**

12. **The cheap corner wins.** On SCAN-B, sub-second GPU filters reach AUC
    0.97. Nothing slower is clearly better: the established leads of tree
    importance are under 0.01 AUC (point 8). On 16 splits, Berk-Jones at
    0.65 s per fit ties a 200-tree forest at 13.5 s. On BreastMNIST every
    method is cheap. Lasso is best at k = 250.
13. **A GPU helps sorting and matrix products, not everything.** A cheap
    GPU is slow in 64-bit arithmetic and has little memory. The data must
    also be copied to it. RFE stayed faster on the CPU. The random forest
    could not move to the GPU.
14. **Exactness has a price in agreement.** Faster exact solvers can pick
    different genes than scikit-learn's default. Lasso does on SCAN-B. RFE
    does on some synthetic data, and in 14 of 100 cells of a preliminary
    SCAN-B frontier run (section 8.2). The reason is that the default
    stops before it reaches the solution. AUC did not change. So RFE, RFECV and
    Lasso now run only their exact solvers (section 6.2).
15. **Selection buys a smaller model more than a better one.** On
    BreastMNIST, all 2,048 features gave AUC 0.921, more than any selected
    subset. On FGVC-Aircraft, all features gave 0.864, also the best. On
    SCAN-B, the best selected sets beat all 28,552 genes by only 0.006 on
    average (0.025 on LumB). They use a tenth of the genes or fewer. On 16
    splits, the corrected test establishes no gain over all genes on
    either dataset. So decide first whether you want a smaller model or a
    better one.

## 9.3 What this project adds

Section 1.7 lists the contributions. In short:

- A method comparison on a cohort of 2,976 patients, against a random
  control, with paired tests corrected for overlapping splits. The
  correction removed two conclusions that an uncorrected test
  supported.
- Evidence that Higher Criticism and Berk-Jones, used as selectors on
  data with widespread signal, reduce to a fixed top-10% filter.
- A cost-quality frontier over every method and setting.
- A package of 16 drop-in selectors, tested against scikit-learn, with
  GPU paths and exact solvers.
- Data checks that changed how the data was used: repeated patients,
  computed labels, and copied images found in BreastMNIST itself.

## 9.4 Answer to the main question

For a given number of features `k`, which method gives the best
classifier, and by how much over random?

On SCAN-B, tree importance, Mann-Whitney and Lasso form a top group. Their
order is not established. Tree importance has the highest median. Its gain
over random is 0.29 macro-F1 at k = 5 and falls to 0.03 at k = 1000.
Mann-Whitney is within 0.03 of it and far more stable. The gain over
random is established up to k = 100. On the frontier, scored by AUC over
16 splits, tree importance's lead over Mann-Whitney is established at
k = 250 only. That lead is 0.007.

On BreastMNIST, the top group is Lasso and tree importance. Their order is
not established either. In the median, Lasso beats random by 0.22 at k = 5
and by 0.07 at k = 250.

If time also counts, a Mann-Whitney U filter on a GPU is the practical
choice for gene expression. It comes within a hundredth of the best AUC,
in under a second.
