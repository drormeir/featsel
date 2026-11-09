Status: in progress

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

## Table of contents

| # | Chapter | What you will learn |
|---|---|---|
| 1 | [The problem](01_the_problem.md) | What feature selection is, and why many features and few samples make it hard. The four tasks of the project. |
| 2 | [The data](02_the_data.md) | SCAN-B gene expression and BreastMNIST images. Where the labels come from, and the traps we found in the data. |
| 3 | [How to measure a selection method](03_how_to_measure.md) | How to test a method without fooling yourself: leakage, splits, the random control, metrics, stability. |
| 4 | [The methods](04_the_methods.md) | The 15 methods, grouped into filters, embedded methods and wrappers, with Higher Criticism. Why we score genes with the U test. |
| 5 | [The software](05_the_software.md) | How `featsel` is built: drop-in selectors, the config-driven runner, datasets, memory safety, reproducibility. |
| 6 | [Making it fast](06_making_it_fast.md) | Task 3: the runner against plain scikit-learn, and faster versions of each method, on the CPU and on a GPU. |
| 7 | [Results: which method selects best](07_results_quality.md) | The comparison on both datasets: gain over random, stability, Higher Criticism, the PAM50 genes. |
| 8 | [Results: cost against quality](08_results_frontier.md) | Every method on one plot of time against AUC, on all five tasks and on each task. |
| 9 | [Conclusions](09_conclusions.md) | What we learned, in plain sentences. |
| 10 | [Open work and next steps](10_open_work.md) | What is missing in this project, and what is worth investigating beyond it. |
| A | [Decision log](appendix_a_decisions.md) | Every important decision, its date, its reason and the alternatives we rejected. |
| B | [Glossary](appendix_b_glossary.md) | Short definitions of the terms used in the report. |
| C | [How to reproduce](appendix_c_reproduce.md) | The commands, configs and notebooks behind every number. |
| | [References](references.md) | The papers cited. |

## How to read the numbers

Every number in this report comes from a file in the repository. Each table
names its source. The two main experiments differ, and the report keeps
them apart:

- **The comparison** (chapter 7): 7 methods, 16 random splits, half the
  patients for training, scored by macro-F1, run 2026-10-31.
- **The frontier** (chapter 8): 15 method settings, 5 random splits, 70% of
  the patients for training, scored by ROC AUC, with timing, run 2026-11-09.

Times are from one laptop: Intel i7-9750H (6 cores, 12 threads), 16 GB RAM,
NVIDIA GeForce GTX 1650 with 4 GB. A time means something only on the
machine it was taken on.

## Related material

- The earlier working drafts, with generated tables: [`docs/report/`](../report/).
- Design notes: [`docs/experiment_design.md`](../experiment_design.md),
  [`docs/architecture_options.md`](../architecture_options.md).
- The project specification: [`references/MLDS_Feature_Selection.pdf`](../../references/MLDS_Feature_Selection.pdf).
