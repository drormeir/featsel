Status: first draft complete

# Choosing Few Genes from Many: Feature Selection for High-Dimensional Data

M.Sc. final project, Machine Learning and Data Science, Reichman University.
Supervisor: Dr. Ben Galili.

This report is written for a reader with a B.Sc.-level background in
machine learning. It explains the problem first, then the methods we tried,
then what we found. Each chapter is short and can be read alone.

## In one paragraph

A gene expression dataset describes each patient by about 30,000 numbers,
one per gene, but holds only a few thousand patients. A classifier trained
on all the genes is slow and can learn noise. Feature selection picks a
small set of genes that still predicts well. We built `featsel`, a Python
package of 15 feature selection methods that plug into scikit-learn. We
compared the methods on breast cancer data (SCAN-B, 2,976 patients, 28,552
genes) and on image data (BreastMNIST, 780 images, 2,048 features), against
a control that picks features at random. We also made the slow methods
faster, many of them on a GPU, and measured cost against quality on one
plot. The main finding: a simple one-gene-at-a-time test is as good as the
complex methods here, and much faster.

## Key results at a glance

| Question | Answer | Where |
|---|---|---|
| Does selection beat random genes? | Yes, by 0.26 to 0.29 macro-F1 at 5 genes on SCAN-B; the gain shrinks to a few hundredths by 250 genes. | [7.2](07_results_quality.md#72-scan-b) |
| Which method selects best? | Tree importance on gene expression, Lasso on image embeddings; Mann-Whitney is close behind and by far the most stable. | [7.2](07_results_quality.md#72-scan-b), [7.8](07_results_quality.md#78-breastmnist) |
| Which is the best value for time? | On SCAN-B, a GPU Mann-Whitney or Higher Criticism filter: AUC 0.97 in 0.66 s. Nothing slower is clearly better. | [8.3](08_results_frontier.md#83-scan-b-all-five-tasks) |
| Does Higher Criticism pick its own count? | No: on both datasets it stopped at the edge of its search, because the signal is neither rare nor independent. | [7.5](07_results_quality.md#75-higher-criticism-did-not-choose-its-own-count) |
| Do methods find the genes that define the label? | Rarely at small `k`, yet they predict it well: correlated genes stand in for them. | [7.6](07_results_quality.md#76-does-selection-find-the-pam50-genes) |
| How much faster did we make it? | The runner 3.2 to 5.3 times over plain scikit-learn; GPU filters 2.6 to 49 times; Sequential selection 39 times; same features. | [6](06_making_it_fast.md) |
| Why did RFECV fail, and can it be fixed? | Its default inner score, accuracy, made it keep 1 or 2 genes on imbalanced tasks. Scored by AUC it recovers to 0.961 on SCAN-B, still below a sub-second filter. | [8.6](08_results_frontier.md#86-rfecv-testing-the-hypothesis) |
| Does selection beat using every feature? | Barely. On SCAN-B the best selected genes beat all 28,552 by 0.005 AUC (0.024 on LumB). On BreastMNIST all 2,048 features score best (0.921). The main gain is a much smaller model. | [8.7](08_results_frontier.md#87-is-selection-needed-at-all) |

## Table of contents

| # | Chapter | What you will learn |
|---|---|---|
| 1 | [The problem](01_the_problem.md) | What feature selection is, and why many features and few samples make it hard. The four tasks of the project. What earlier studies found. |
| 2 | [The data](02_the_data.md) | SCAN-B gene expression and BreastMNIST images. Where the labels come from, and the traps we found in the data. |
| 3 | [How to measure a selection method](03_how_to_measure.md) | How to test a method without fooling yourself: leakage, splits, the random control, metrics, stability. |
| 4 | [The methods](04_the_methods.md) | The 15 methods, grouped into filters, embedded methods and wrappers, with Higher Criticism. Why we score genes with the U test. |
| 5 | [The software](05_the_software.md) | How `featsel` is built: drop-in selectors, the config-driven runner, datasets, memory safety, reproducibility. |
| 6 | [Making it fast](06_making_it_fast.md) | Task 3: the runner against plain scikit-learn, and faster versions of each method, on the CPU and on a GPU. |
| 7 | [Results: which method selects best](07_results_quality.md) | The comparison on both datasets: gain over random, stability, Higher Criticism, the PAM50 genes, and what it means. |
| 8 | [Results: cost against quality](08_results_frontier.md) | Every method on one plot of time against AUC, on all five tasks and on each task. |
| 9 | [Conclusions](09_conclusions.md) | What we learned, in plain sentences. |
| 10 | [Open work and next steps](10_open_work.md) | What is missing in this project, and what is worth investigating beyond it. |
| A | [Decision log](appendix_a_decisions.md) | Every important decision, its date, its reason and the alternatives we rejected. |
| B | [Glossary](appendix_b_glossary.md) | Short definitions of the terms used in the report. |
| C | [How to reproduce](appendix_c_reproduce.md) | The commands, configs and notebooks behind every number. |
| D | [Data: plans and open questions](appendix_d_data_details.md) | Reproducing the ER label, the PAM50 recovery design, Higher Criticism's share of signal, a better model for floor-heavy genes. |
| E | [Full comparison tables](appendix_e_comparison_tables.md) | Per-task scores at every `k`, genes kept, times, stability and chosen C. |
| | [References](references.md) | The papers cited. |

## How to read the numbers

Every number in this report comes from a file in the repository. Each table
names its source. The two main experiments differ, and the report keeps
them apart:

- **The comparison** (chapter 7): 7 methods, 16 random splits, half the
  patients for training, scored by macro-F1, run 2026-10-31.
- **The frontier** (chapter 8): 13 method settings, 5 random splits, 70% of
  the patients for training, scored by ROC AUC, with timing, run 2026-11-09.

Times are from one laptop: Intel i7-9750H (6 cores, 12 threads), 16 GB RAM,
NVIDIA GeForce GTX 1650 with 4 GB. A time means something only on the
machine it was taken on.

## Related material

- This report is the project's only design record. The earlier drafts
  (`docs/report/`) and working notes (`docs/*.md`) were merged into it and
  removed; git history keeps them. Decisions are in
  [Appendix A](appendix_a_decisions.md), parked ideas in
  [section 10.3](10_open_work.md#103-parked-ideas).
- The project specification: [`references/MLDS_Feature_Selection.pdf`](../../references/MLDS_Feature_Selection.pdf).
