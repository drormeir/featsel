# Choosing Few Genes from Many: Feature Selection for High-Dimensional Data

M.Sc. final project, Machine Learning and Data Science, Reichman University.
Supervisor: Dr. Ben Galili.

The report is for a reader with a B.Sc.-level background in machine
learning. Each chapter is short and can be read alone.

## In one paragraph

A gene expression dataset describes each patient by about 30,000 numbers,
one per gene, but holds only a few thousand patients. A classifier trained
on all the genes is slow and can learn noise. Feature selection picks a
small set of genes that still predicts well. We built `featsel`, a Python
package of 16 selection methods that plug into scikit-learn. We compared
them, against a control that picks features at random, on breast cancer
data (SCAN-B, 2,976 patients, 28,541 genes) and image data (BreastMNIST,
780 images, 2,048 features). Aircraft photos then tested the finished
pipeline on data it was not designed for. We made the slow methods faster,
many on a GPU, and plotted cost against quality. The main finding: here, a
simple test of one gene at a time is almost as good as the complex
methods, and much faster.

## Key results at a glance

The image-data answers come from the rerun of 2026-11-14 (chapter 8). The
SCAN-B answers still come from runs made before 2026-11-13. Since then, RFE,
RFECV and Lasso use only exact solvers. Every method now chooses its
number of features on validation sets, not by test-set AUC (sections 3.11
and 6.2). The results will be updated after the rerun.

| Question | Answer | Where |
|---|---|---|
| Does selection beat random genes? | Yes: by 0.26 to 0.29 macro-F1 at 5 genes on SCAN-B. The gain shrinks to a few hundredths by 250 genes. | [7.2](07_results_quality.md#72-scan-b) |
| Which method selects best? | No single method. Tree importance, Mann-Whitney and Lasso form a top group. A paired test corrected for overlapping splits cannot order them. Mann-Whitney is by far the most stable. | [7.2](07_results_quality.md#72-scan-b), [7.3](07_results_quality.md#73-are-the-differences-real-paired-tests), [7.8](07_results_quality.md#78-breastmnist) |
| Which is the best value for time? | On SCAN-B, a GPU Mann-Whitney or Higher Criticism filter: AUC 0.97 in 0.66 s. Nothing slower is clearly better. | [8.3](08_results_frontier.md#83-scan-b-all-five-tasks) |
| Does Higher Criticism pick its own count? | No. On all three datasets it stopped at the edge of its search: the signal is neither rare nor independent. Berk-Jones, its exact relative, stopped at or just before that edge on the two datasets it ran on. | [7.5](07_results_quality.md#75-higher-criticism-did-not-choose-its-own-count), [8.8](08_results_frontier.md#88-an-unseen-dataset-fgvc-aircraft) |
| Do methods find the genes that define the label? | Rarely at small `k`. Yet they predict the label well, because correlated genes stand in for them. | [7.6](07_results_quality.md#76-does-selection-find-the-pam50-genes) |
| How much faster did we make it? | The runner: 3.2 to 5.3 times faster than plain scikit-learn. GPU filters: 2.6 to 49 times. Sequential selection: 34 times. All select the same features as before. | [6](06_making_it_fast.md) |
| Why did RFECV fail, and can it be fixed? | Its default validation score, accuracy, made it keep 1 or 2 genes on imbalanced tasks. Scored by AUC, it recovers to 0.961 on SCAN-B. That is still below a sub-second filter. | [8.6](08_results_frontier.md#86-rfecv-testing-the-hypothesis) |
| Does selection beat using every feature? | Not on image data: on BreastMNIST all 2,048 features score 0.924 and RFE ties them (0.923); on FGVC-Aircraft both score 0.862. Selection's gain there is a smaller model. SCAN-B (earlier run): the best selected genes beat all 28,552 by 0.006 AUC, not established by the corrected test. | [8.3](08_results_images.md#83-selection-against-all-features), [8.7](08_results_frontier.md#87-is-selection-needed-at-all) |
| Does the pipeline work on unseen data? | Yes. It ran unchanged, entirely on the GPU, on aircraft photos. At 10 features, Mann-Whitney beat random by 0.124 AUC and RFE by 0.103, on all 16 splits. | [8.9](08_results_images.md#89-an-unseen-dataset-fgvc-aircraft) |

## Table of contents

| # | Chapter | Contents |
|---|---|---|
| 1 | [The problem](01_the_problem.md) | Feature selection, why few samples and many features make it hard, the four tasks, earlier studies. |
| 2 | [The data](02_the_data.md) | SCAN-B and BreastMNIST: where the labels come from, and the traps we found. |
| 3 | [How to measure a selection method](03_how_to_measure.md) | Testing a method without fooling yourself: leakage, splits, the random control, metrics, stability. |
| 4 | [The methods](04_the_methods.md) | The 16 methods: filters, embedded methods, wrappers, Higher Criticism and Berk-Jones. Why genes are scored with the U test. |
| 5 | [The software](05_the_software.md) | Drop-in selectors, the config-driven runner, datasets, memory safety, reproducibility. |
| 6 | [Making it fast](06_making_it_fast.md) | Task 3: the runner against plain scikit-learn; faster methods on the CPU and GPU. |
| 7 | [Results: SCAN-B](07_results_quality.md) | Being rebuilt from the rerun. Until then it reports an earlier comparison. |
| 8 | [Results on image data](08_results_images.md) | BreastMNIST and FGVC-Aircraft: AUC at the chosen `k`, gain over random, all features, stability, Higher Criticism, chains. |
| — | [Earlier frontier run](08_results_frontier.md) | Being replaced: its SCAN-B parts move to chapter 7, its image parts are superseded by chapter 8. |
| 9 | [Conclusions](09_conclusions.md) | What we learned, in plain sentences. |
| 10 | [Open work and next steps](10_open_work.md) | What is missing, and what is worth investigating next. |
| A | [Glossary](appendix_a_glossary.md) | Short definitions of terms. |
| B | [How to reproduce](appendix_b_reproduce.md) | The commands, configs and notebooks behind every number. |
| C | [Data: plans and open questions](appendix_c_data_details.md) | Reproducing the ER label, the PAM50 recovery design, Higher Criticism's share of signal, a better model for floor-heavy genes. |
| D | [Full comparison tables](appendix_d_comparison_tables.md) | Per-task scores at every `k`, genes kept, times, stability and chosen C. |
| | [References](references.md) | The papers cited. |

## How to read the numbers

Every number comes from a file in the repository, and each table names its
source. Every experiment follows one protocol (chapter 3): 16 random
splits, 70% of the samples in the training set, `k` chosen on validation
sets, ROC AUC, and timing. Chapter 8 (image data) comes from the rerun
under this protocol. Until the SCAN-B rerun finishes, chapter 7 and the
earlier frontier file report earlier runs under older protocols.

Times are from one laptop: Intel i7-9750H (6 cores, 12 threads), 16 GB
RAM, NVIDIA GeForce GTX 1650 with 4 GB. A time is meaningful only on the
machine that measured it.

## Related material

- The project specification: [`references/MLDS_Feature_Selection.pdf`](../references/MLDS_Feature_Selection.pdf).
