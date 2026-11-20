# Choosing Few Genes from Many: Feature Selection for High-Dimensional Data

OUTLINE, not the report text. Each section gives its page budget and the
claims it keeps. "D" means the detailed report
([`detailed_report/`](../detailed_report/README.md)); "D 7.2" is its
section 7.2. Every number in the report comes from a file named in D.

Budget: 18 pages, appendices included. About 8,000 words, 2 figures,
1 results table, a few short tables.

---

## Summary (1 page)

- The question: for a budget of `k` features, which selection method gives
  the best classifier, and by how much does it beat `k` random features?
  (D 1.4)
- What was built: `featsel`, 16 selectors that drop into scikit-learn,
  and a runner driven by one config file (D 9.1).
- The data: SCAN-B gene expression (2,976 patients, 28,541 genes, five
  PAM50 tasks), BreastMNIST embeddings, and FGVC-Aircraft embeddings,
  unseen during the design (D 2.4).
- The answer: at 10 features, selection beats random by 0.10 to 0.20 AUC.
  At the chosen `k`, the top methods cannot be told apart. Mann-Whitney is
  within 0.01 of the best on all three datasets, in under a second on a
  GPU (D 9.3 items 4, 7, 8).
- Selection rarely beats all the features: never on the image data, on
  one SCAN-B task of five (D 9.3 item 16).
- Speed: the GPU filters run 2.6 to 49 times faster (D 6.2).

## 1. Introduction (1.5 pages)

- The p ≫ n problem, in four sentences: overfitting, false discoveries,
  correlated features, cost (D 1.2).
- The four tasks of the specification (D 1.3).
- The main question and the cost question (D 1.4).
- Four known pitfalls and the answer to each, as a table (D 1.6).
- What this project adds: five items (D 1.7).

## 2. Data (1.5 pages)

- One table: the three datasets, their samples, features, tasks, label
  source and signal (D 2.4).
- SCAN-B choices, in one table: HiSeq only; one sample per patient,
  because 93 patients appear twice and would leak; flat genes dropped
  (D 2.1).
- Only the order of a gene's values means something, and about half the
  values sit at a floor. So genes are scored by the U test (D 2.1, D 4.3).
- PAM50 is computed from 50 known genes. Prediction partly re-learns a
  published classifier, and the right genes are known (D 2.1).
- Five binary tasks, one per subtype, with their class sizes (D 2.1).
- The signal is widespread and correlated: more than 10% of genes carry
  it (D 2.1).
- BreastMNIST: 150 groups of copied images, found by the loader (D 2.2).
- FGVC-Aircraft: ResNet-50 V1 weights, not V2: AUC 0.862 against 0.742
  (D 2.3, `results/aircraft_weights.csv`).

## 3. Protocol (1.5 pages)

- One table: splits, main score, control, choice of `k`, classifier,
  timing (D 3 intro).
- Selection inside each split. On pure noise, selection on all the data
  gave AUC 0.96 instead of 0.5 (D 3.2).
- 16 stratified Monte Carlo splits, not k-fold; why 16 is enough (D 3.3).
- The random control at every `k` (D 3.4).
- ROC AUC as the main score; why not accuracy (D 3.5).
- The corrected resampled t-test for overlapping splits (D 3.3).
- Stability by Nogueira's measure (D 3.6).
- `k` chosen on validation sets inside the training set (D 3.11).

## 4. Methods (2 pages)

- The three families, as a table (D 4.1).
- The 16 methods: one table, method, family and idea (D 4.2).
- Why the U test, not the F-test (D 4.3).
- Higher Criticism: the idea; our one change, i/(N+1); search kept at
  10%, so as not to tune to the answer (D 4.4).
- Berk-Jones: a test used here as a selector (D 4.5).
- Which methods the experiments ran (D 4.6).

## 5. Software and speed (2 pages)

- Two deliverables: drop-in selectors and a config-driven runner. One
  diagram (D 5.1).
- The selector contract: a scikit-learn estimator; tested to pick the
  same features as scikit-learn, and to be at most 5% slower (D 5.2, 5.6).
- The runner: rank once, cut every `k`; memory safety and resume, in one
  paragraph (D 5.4, 5.5).
- The runner against plain scikit-learn: 3.2 to 5.3 times faster, with
  identical scores. Measured before liblinear was dropped (D 6.1).
- Parallel work gains at most 1.8 times on this laptop; memory bandwidth
  is the likely limit (D 6.1).
- Each method, faster: one short table, GPU speed-ups (D 6.2).
- Exact solvers: liblinear stops before the solution. RFE became 8 times
  faster and repeatable, at the same AUC (D 6.2).
- Why the GPU does not always win: 64-bit speed, copying, 4 GB (D 6.2).

## 6. Results (4 pages)

- Figure 1: cost against quality, one panel per dataset
  (`scripts/plot_final_frontier.py`, from D 7.1, 8.1, 8.9).

![Figure 1](figures/frontier.png)

- Figure 2: test AUC against `k`, one panel per dataset; SCAN-B as the
  mean of its five tasks (`scripts/plot_curves.py --mean --width 7.2`, from D 7.2, 8.2).

![Figure 2](figures/curves.png)

- Table 1: each method at its chosen `k`. SCAN-B values are the mean of
  its five tasks; time is the mean of each task's median
  (`scripts/final_table.py`, `figures/table1.csv`).

| Method | SCAN-B AUC | BreastMNIST AUC | FGVC-Aircraft AUC | SCAN-B selection, s per fit | SCAN-B stability |
|---|---|---|---|---|---|
| Mutual information | 0.971 | 0.916 | 0.847 | 3.6 | 0.70 |
| Berk-Jones | 0.970 | 0.867 | — | 0.66 | 0.92 |
| HC2008 | 0.970 | 0.868 | 0.817 | 0.68 | 0.92 |
| Correlation (Spearman) | 0.969 | 0.905 | 0.824 | 3.6 | 0.78 |
| Mann-Whitney | 0.969 | 0.918 | 0.854 | 0.67 | 0.76 |
| Tree importance | 0.969 | 0.906 | — | 17 | 0.27 |
| Mann-Whitney, then Lasso | 0.968 | 0.878 | — | 0.79 | 0.46 |
| Mann-Whitney, then tree importance | 0.965 | 0.879 | — | 2.8 | 0.51 |
| Lasso | 0.964 | 0.898 | 0.846 | 0.59 | 0.36 |
| All features (no selection) | 0.963 | 0.924 | 0.862 | — | — |
| Mann-Whitney, then RFE | 0.962 | 0.889 | — | 3.0 | 0.42 |
| RFECV, scored by AUC | 0.962 | 0.912 | 0.858 | 5.7 | 0.23 |
| RFE | 0.961 | 0.923 | 0.862 | 6.9 | 0.33 |
| HC2004 | 0.874 | 0.758 | 0.674 | 0.68 | 0.67 |
| RFECV, scored by accuracy | 0.720 | 0.889 | — | 6.3 | 0.20 |

- 6.1 Selection against random: large gain at 10 features, small at
  2,000; LumA and LumB keep about 0.02 (D 7.2, 8.2).
- 6.2 Selection against all features: a gain only on LumB; none on the
  image data (D 7.3, 8.3).
- 6.3 No single best method; stability separates them (D 7.1, 7.4, 8.4).
- 6.4 Higher Criticism and Berk-Jones stop at the edge of their search
  (D 7.5, 8.5).
- 6.5 The PAM50 genes: few found in the top 50, yet good prediction
  (D 7.6).
- 6.6 Wrappers: RFECV needs AUC as its inner score; a filter first
  rescues RFE at 10 genes (D 7.8, 7.9).
- 6.7 Without the copied images: about 0.03 AUC lower, same order
  (D 8.8).
- 6.8 Cost: a GPU filter is the best value; choosing `k` costs more than
  one fit (D 7.11).

## 7. Discussion (2 pages)

- The answer to the main question (D 9.5).
- What holds on every dataset, and what changes (D 9.4).
- Key decisions: one table, decision, reason, effect (D 9.2).
- Limits (D 9.6).
- Open research questions: a rare-signal dataset for Higher Criticism;
  an exact test for floor-heavy genes; how often random sets beat a
  chosen set; correlated genes as one unit; stability selection; a
  measured target such as ER (D 10).

## References (1 page)

- Only works cited in this report (from D references).

## Appendix A. How to reproduce (1 page)

- Setup, the experiment commands, determinism, tests (D A).

## Appendix B. Engineering notes (1 page)

- Lasso's solver: why PPDNA, and its changes from the paper (D D.3).
- How the timings were checked (D D.1).
- The forest on a 4 GB GPU (D D.5).

---

Left out, kept in D: the classifier study (D 7.7), side scores and
memory (D 7.12, 8.10), Sequential selection in detail (D 7.10), the
per-task plots (D 7.1), data precision (D 3.12), the cost formulas
(D 4.7), the loader and rejected designs (D 5.2, 5.3), the full tables
(D C), the data plans (D B) and the glossary (D E).
