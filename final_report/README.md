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

| Question | Answer | Where |
|---|---|---|
| Does selection beat random genes? | Yes, most at small `k`: at 10 features by 0.10 to 0.20 AUC on SCAN-B and 0.10 to 0.16 on image data. The gain shrinks as `k` grows; on LumA and LumB it lasts to 2,000 genes. | [7.2](07_results_scanb.md#72-the-bar-random-genes-at-each-k), [8.2](08_results_images.md#82-the-bar-random-features-at-each-k) |
| Which method selects best? | No single method. On SCAN-B eleven methods lie within 0.01 mean AUC, and none leads on every task. On image data RFE ties all features. Mann-Whitney is within 0.01 of the best everywhere and among the most stable. | [7.1](07_results_scanb.md#71-what-each-method-reaches), [7.4](07_results_scanb.md#74-stability), [8.1](08_results_images.md#81-breastmnist-what-each-method-reaches) |
| Which is the best value for time? | A GPU filter. On SCAN-B, HC2008 and Berk-Jones reach 0.970 mean AUC in 0.66 s per fit; the best, mutual information, reaches 0.971 in 3.6 s. | [7.11](07_results_scanb.md#711-cost) |
| Can the best method for a budget be known in advance? | No. RFE leads at large `k` on image data and trails on SCAN-B; at 10 features filters lead on image data, Mann-Whitney then RFE on SCAN-B. | [9.3](09_conclusions.md#93-across-the-datasets) |
| Does Higher Criticism pick its own count? | No. On all three datasets it stopped at the edge of its search, 10% of the features: the signal is not rare. Berk-Jones stopped at or just before that edge. | [7.5](07_results_scanb.md#75-higher-criticism-and-berk-jones), [8.5](08_results_images.md#85-higher-criticism-and-berk-jones) |
| How much faster did we make it? | The runner: 3.2 to 5.3 times faster than plain scikit-learn. GPU filters: 2.6 to 49 times. Sequential selection: 34 times. All select the same features as before. | [6](06_making_it_fast.md) |
| Why did RFECV fail, and can it be fixed? | Its default validation score, accuracy, made it keep one gene on three imbalanced tasks. Scored by AUC, it recovers to 0.962 on SCAN-B, level with RFE but below a sub-second filter. | [7.8](07_results_scanb.md#78-rfecv-accuracy-against-auc) |
| Does selection beat using every feature? | Not on image data: all 2,048 features score best (BreastMNIST 0.924, FGVC-Aircraft 0.862) and RFE ties them. On SCAN-B, ten methods beat all 28,541 genes on LumB by 0.014 to 0.023; on the other tasks no gain is established. | [7.3](07_results_scanb.md#73-selection-against-all-genes), [8.3](08_results_images.md#83-selection-against-all-features) |
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
| 7 | [Results on SCAN-B](07_results_scanb.md) | AUC per task at the chosen `k`, gain over random, all genes, stability, Higher Criticism, RFECV, chains, Sequential, cost. |
| 8 | [Results on image data](08_results_images.md) | BreastMNIST and FGVC-Aircraft: AUC at the chosen `k`, gain over random, all features, stability, Higher Criticism, chains. |
| 9 | [Conclusions](09_conclusions.md) | What we learned, what the datasets share, and the answer to the main question. |
| 10 | [Open work and next steps](10_open_work.md) | What is missing, and what is worth investigating next. |
| A | [Glossary](appendix_a_glossary.md) | Short definitions of terms. |
| B | [How to reproduce](appendix_b_reproduce.md) | The commands, configs and notebooks behind every number. |
| C | [Data: plans and open questions](appendix_c_data_details.md) | Reproducing the ER label, the PAM50 recovery design, Higher Criticism's share of signal, a better model for floor-heavy genes. |
| D | [Full tables](appendix_d_comparison_tables.md) | Test AUC at every `k` and at the chosen `k`, per task and dataset, with genes kept, times and stability. |
| | [References](references.md) | The papers cited. |

## How to read the numbers

Every number comes from a file in the repository, and each table names its
source. Every experiment follows one protocol (chapter 3): 16 random
splits, 70% of the samples in the training set, `k` chosen on validation
sets, ROC AUC, and timing. Chapters 7 and 8 come from one rerun under
this protocol, made on 2026-11-14 and 15.

Times are from one laptop: Intel i7-9750H (6 cores, 12 threads), 16 GB
RAM, NVIDIA GeForce GTX 1650 with 4 GB. A time is meaningful only on the
machine that measured it.

## Related material

- The project specification: [`references/MLDS_Feature_Selection.pdf`](../references/MLDS_Feature_Selection.pdf).
