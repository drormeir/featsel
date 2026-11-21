# Choosing Few Genes from Many: Feature Selection for High-Dimensional Data

M.Sc. final project, Machine Learning and Data Science, Reichman University.
Supervisor: Dr. Ben Galili.

Each chapter is short and can be read alone. For the question, the
methods and the main results in one page, read the
[abstract](abstract.md).

## What was built

- **A library of 16 feature selectors** that plug into scikit-learn. The
  ones that copy a scikit-learn method are tested to select the same
  features. All of these but Lasso are also tested to be at most 5% slower
  (chapter 5).
- **GPU paths for 14 selectors**, in PyTorch. Ten filters share one rule
  for choosing the device (section 6.3), and select the same features as
  on the CPU. Lasso, RFE, RFECV and Sequential selection can also run on
  a GPU. There the filters run 2.6 to 49 times faster than the originals they replace, and sequential
  selection 34 times (chapter 6).
- **Exact solvers** for RFE, RFECV and Lasso, which reach the solution
  that scikit-learn's default solver stops short of. On SCAN-B, one RFE
  fit became 8 times faster, 23.1 s against 2.8 s (chapter 6).
- **An experiment runner driven by one YAML file.** It selects inside
  every split, runs a random control at every `k`, and chooses `k` on
  validation sets. It runs jobs in parallel, survives running out of
  memory, and resumes after a crash. The full SCAN-B comparison took
  about 33 hours (chapter 5, section 7.11).
- **Dataset loaders** that download the public data once, check it, and
  record every choice made on it. Two of these checks changed the study:
  repeated patients in SCAN-B, and copied images in BreastMNIST
  (chapter 2).
- **A statistical protocol** with a test corrected for overlapping splits.
  It removed most of the claimed differences from all genes (chapter 3,
  section 9.2).

## Key results at a glance

| Question | Answer | Where |
|---|---|---|
| Does selection beat random genes? | Yes, most at small `k`. At 10 features, the filters, tree importance, the chains and Sequential beat random by 0.11 to 0.20 AUC on SCAN-B. On image data, the filters and tree importance beat it by 0.10 to 0.16. The gain shrinks as `k` grows. On LumA and LumB, about 0.02 remains at 2,000 genes. | [7.2](07_results_scanb.md#72-the-bar-random-genes-at-each-k), [8.2](08_results_images.md#82-the-bar-random-features-at-each-k) |
| Which method selects best? | No single method. On SCAN-B eleven methods lie within 0.01 mean AUC, and none leads on every task. On image data RFE ties all features. Mann-Whitney is within 0.012 of the best everywhere and among the most stable. | [7.1](07_results_scanb.md#71-what-each-method-reaches), [7.4](07_results_scanb.md#74-stability), [8.1](08_results_images.md#81-breastmnist-what-each-method-reaches) |
| Which is the best value for time? | A GPU filter. On SCAN-B, HC2008 and Berk-Jones reach 0.970 mean AUC in 0.66 to 0.68 s per fit; the best, mutual information, reaches 0.971 in 3.6 s. | [7.11](07_results_scanb.md#711-cost) |
| Can the best method for a budget be known in advance? | No. RFE leads at large `k` on image data (established on FGVC-Aircraft) and trails on SCAN-B; at 10 features filters and tree importance, alone or chained, lead on image data, Mann-Whitney then RFE on SCAN-B. | [9.4](09_conclusions.md#94-across-the-datasets) |
| Does Higher Criticism pick its own count? | Not on the real data. On all three datasets HC2008 stopped at the edge of its search, 10% of the features: the signal is not rare. Berk-Jones stopped at or just before that edge. On simulated rare signal, HC2008 kept a median of 33.5 features for 40 informative ones, with the best mean AUC, but not established above Mann-Whitney. | [7.5](07_results_scanb.md#75-higher-criticism-and-berk-jones), [8.5](08_results_images.md#85-higher-criticism-and-berk-jones), [8.11](08_results_images.md#811-simulated-rare-signal) |
| Do methods find the genes that define the label? | Rarely at small `k`: at most 12% of the 50 PAM50 genes in the top 50, on average. Yet they predict the label well, because correlated genes stand in for them. | [7.6](07_results_scanb.md#76-does-selection-find-the-pam50-genes) |
| How much faster did we make it? | The runner: 4.3 to 10.6 times faster than plain scikit-learn on one core (6.8 times against it on 12 threads). GPU filters: 2.6 to 49 times faster than the originals. Sequential selection: 34 times. These select the same features as before. | [6](06_making_it_fast.md) |
| Why did RFECV fail, and can it be fixed? | Its default validation score, accuracy, made it keep one gene on three imbalanced tasks. Scored by AUC, it recovers to 0.962 on SCAN-B, level with RFE but below a sub-second filter. | [7.8](07_results_scanb.md#78-rfecv-accuracy-against-auc) |
| Does selection beat using every feature? | Not on image data: all 2,048 features score best (BreastMNIST 0.895, FGVC-Aircraft 0.862) and RFE ties them. On SCAN-B, ten methods beat all 28,541 genes on LumB by 0.014 to 0.023; on the other tasks no gain is established. | [7.3](07_results_scanb.md#73-selection-against-all-genes), [8.3](08_results_images.md#83-selection-against-all-features) |
| Does the pipeline work on unseen data? | Yes. It ran unchanged on aircraft photos, on the GPU except tree importance. At 10 features, Mann-Whitney beat random by 0.124 AUC and RFE by 0.103, on all 16 splits. | [8.9](08_results_images.md#89-an-unseen-dataset-fgvc-aircraft) |

## Table of contents

| # | Chapter | Contents |
|---|---|---|
| | [Abstract](abstract.md) | The question, the methods, the results and the conclusions, in one page. |
| 1 | [The problem](01_the_problem.md) | Feature selection, why few samples and many features make it hard, the four tasks, earlier studies. |
| 2 | [The data](02_the_data.md) | SCAN-B, BreastMNIST and FGVC-Aircraft: where the labels come from, and the traps we found. |
| 3 | [How to measure a selection method](03_how_to_measure.md) | Testing a method without fooling yourself: leakage, splits, the random control, metrics, stability. |
| 4 | [The methods](04_the_methods.md) | The 16 methods: filters, embedded methods, wrappers, Higher Criticism and Berk-Jones. Why genes are scored with the U test. |
| 5 | [The software](05_the_software.md) | Drop-in selectors, the config-driven runner, datasets, memory safety, reproducibility. |
| 6 | [Making it fast](06_making_it_fast.md) | Task 3: the runner against plain scikit-learn; faster methods on the CPU and GPU. |
| 7 | [Results on SCAN-B](07_results_scanb.md) | AUC per task at the chosen `k`, gain over random, all genes, stability, Higher Criticism, RFECV, chains, Sequential, cost, side scores and memory. |
| 8 | [Results on image data](08_results_images.md) | BreastMNIST and FGVC-Aircraft: AUC at the chosen `k`, gain over random, all features, stability, Higher Criticism, chains, side scores and memory. Simulated rare signal for Higher Criticism. |
| 9 | [Conclusions](09_conclusions.md) | What we built, the key decisions, what we learned, what the datasets share, the answer to the main question, and the limits. |
| 10 | [Open work and next steps](10_open_work.md) | What is missing, and what is worth investigating next. |
| A | [How to reproduce](appendix_a_reproduce.md) | The commands, configs and notebooks behind every number. |
| B | [Data: plans, open questions and extra facts](appendix_b_data_details.md) | Reproducing the ER label, the PAM50 recovery design, Higher Criticism's share of signal, a better model for floor-heavy genes, gene counts. |
| C | [Full tables](appendix_c_comparison_tables.md) | Test AUC at every `k` and at the chosen `k`, per task and dataset, with genes kept, times and stability. |
| D | [Engineering notes](appendix_d_engineering.md) | Solver details, timing checks and the GPU trials behind chapter 6. |
| E | [Glossary](appendix_e_glossary.md) | Short definitions of terms. |
| | [References](references.md) | The papers cited. |

## How to read the numbers

Every number comes from a file in the repository, and each table names its
source. The method comparisons follow one protocol (chapter 3): 16
random splits, 70% of the samples in the training set, `k` chosen on
validation sets, ROC AUC, and timing. The main SCAN-B runs were made on
2026-11-14 and 15; later runs, from 2026-11-20 on, among them BreastMNIST
without copies, carry
their dates in their JSON files. A few side studies keep their own
settings, stated where they appear: the PAM50 recovery (section 7.6), the
classifier study (section 7.7) and the speed benchmarks (chapter 6). Section 9.6 gives the limits that
follow.

Times are from one laptop: Intel i7-9750H (6 cores, 12 threads), 16 GB
RAM, NVIDIA GeForce GTX 1650 with 4 GB. A time is meaningful only on the
machine that measured it.

## Related material

- The project specification: [`references/MLDS_Feature_Selection.pdf`](../references/MLDS_Feature_Selection.pdf).
