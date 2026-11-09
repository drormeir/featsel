# 9. Conclusions

[Back to contents](README.md) · Previous: [8. Results: cost against quality](08_results_frontier.md) · Next: [10. Open work](10_open_work.md)

## 9.1 What we built

- **`featsel`**, a Python package of 15 feature selection methods covering
  filters, embedded methods, wrappers, Higher Criticism and a random
  control. Each is a scikit-learn estimator and drops into an existing
  pipeline; the ones that copy a scikit-learn method are tested to pick the
  same features. (Task 1.)
- **A config-driven runner** that compares methods under one protocol:
  selection inside every split, a random control at every `k`, many random
  splits, tuning inside the training part, and one results table. It
  survives memory exhaustion and can resume. (Tasks 1 and 2.)
- **Speed** at two levels: the runner is 3.2 to 5.3 times faster than
  plain scikit-learn with identical scores; nine selectors run on a GPU,
  with the same features as on the CPU, the filters 2.6 to 49 times faster;
  exact solvers make Lasso, RFE, RFECV and Sequential selection faster.
  (Task 3.)
- **A second dataset**, BreastMNIST image embeddings, run through the same
  pipeline unchanged. (Task 4.)

## 9.2 What we learned

**About the data.**

1. Read where the labels come from. PAM50 and the course's ER label are
   computed from the same gene expression we select from. A study that
   treats them as clinical facts overstates what it shows.
2. Check for repeated samples. 93 patients appear twice in the course's
   copy of SCAN-B; random splits put them on both sides and leak.
3. The gene values are on an arbitrary scale, with half the values tied at
   a floor. A rank test (Mann-Whitney U) fits this data; the F-test's
   assumptions do not hold.

**About selection.**

4. **Selection matters most when few features are allowed.** At k = 5 the
   best methods beat random genes by 0.26 to 0.29 macro-F1. By k = 250 the
   gain is a few hundredths, and some methods fall below random at
   k = 1000.
5. **The signal is everywhere.** Thousands of genes carry it, and they move
   together. Random genes catch up fast, and many different gene sets
   predict equally well.
6. **A gene list is therefore not a unique answer.** In their top 50,
   the methods found on average at most 12% of the 50 genes that define
   PAM50 (26% on the best task), while predicting it well. Tree importance scored best and was the least stable.
   Report stability next to accuracy.
7. **Simple is enough.** A one-gene-at-a-time rank test is within 0.03
   macro-F1 of the best method on SCAN-B at every `k`, is by far the most stable,
   and on a GPU takes under a second.
8. **The best method depends on the data.** Tree importance led on gene
   expression; Lasso led on image embeddings. The broad pattern (selection
   beats random most at small `k`; wrappers are weakest) held on both.
9. **Higher Criticism needs rare signal.** On both datasets it stopped at
   the edge of its search range and acted as a fixed top-10% filter. Its
   assumptions, rare and independent signal, do not hold here.
10. **Wrappers did not pay.** RFE, RFECV and Sequential selection were the
    slowest and scored below the simple filters on SCAN-B. RFECV's default
    inner score, accuracy, made it keep a single gene on imbalanced tasks.
    With AUC as its inner score it recovered (0.961 on SCAN-B). A wrapper
    is only as good as its inner score.

**About cost.**

11. **The cheap corner wins.** On SCAN-B, sub-second GPU filters reach AUC
    0.97; nothing slower is clearly better. On BreastMNIST everything is
    cheap; Lasso is best at k ≤ 250.
12. **A GPU helps sorting and matrix products, not everything.** A cheap
    GPU computes in 64 bits slowly, has little memory, and needs the data
    copied to it. RFE stayed faster on the CPU, and the random forest could
    not move to it.
13. **Exactness has a price in agreement.** Faster exact solvers for Lasso
    and RFE pick different genes than scikit-learn's default, because the
    default stops before reaching the solution.
14. **Selection buys a smaller model more than a better one.** On
    BreastMNIST, all 2,048 features gave AUC 0.921, more than any selected
    subset. On SCAN-B, the best selected sets beat all 28,552 genes by only
    0.005 on average (0.024 on LumB), with a tenth of the genes or fewer.
    Know which of the two you want.

## 9.3 Answer to the main question

For a given number of features `k`, which method gives the best
classifier, and by how much over random?

On SCAN-B: tree importance, by 0.29 macro-F1 at k = 5 down to 0.03 at
k = 1000, with Mann-Whitney close behind and far more stable. On
BreastMNIST: Lasso, by 0.22 at k = 5 down to 0.07 at k = 250. If time also
counts, a Mann-Whitney U filter on a GPU is the practical choice on gene
expression: within a hundredth of the best in AUC, in under a second.
