# 10. Open work and next steps

[Back to contents](README.md) · Previous: [9. Conclusions](09_conclusions.md) · Next: [Appendix A: glossary](appendix_a_glossary.md)

Section 10.1 lists unfinished work within the project's four tasks.
Section 10.2 lists research beyond its scope.

## 10.1 Missing work within the project

### Experiments

| Item | Why it matters | Effort |
|---|---|---|
| **Tuned classifier.** C is fixed at 1. | Absolute AUCs may be a little low. A method that needs a different C is at a disadvantage. | About 15 times the classifier's cost. |
| **The full SCAN-B grid**: every selector × classifier × `k` × train fraction. | The protocol uses one classifier and one train fraction. | About four days on this laptop. Skipped by decision. |
| **A measured target on SCAN-B**, such as the pathologist's ER status. | Every SCAN-B task here predicts a computed label. A measured label would test the conclusions against clinical truth. | One config. The loader supports it. |
| **The quantile transform with every classifier** (section 3.10). It did not help logistic regression. | Would show whether shrinkage LDA and k-NN gain from normal-shaped features. | One `k`, all classifiers. A run was started and cancelled. |
| **Time and memory complexity per method**, derived from the algorithm. | To sit next to the measured times. A disagreement would be a finding. | Writing only, no runs. |
| **How SCAN-B computed `pam50 subtype`.** | The GEO notes say only that it implements Parker's method, after centring to a reference set, with unpublished R scripts (section 2.1, Appendix C.2). | Reading other SCAN-B papers. |

### Software

Config validation and other known limits are in section 5.8.

| Item | State |
|---|---|
| **Store each test patient's score**: the true label, the score and the decision threshold. Any metric could then be derived later without a new run. | Designed, not built. Scores differ across classifiers: a linear SVM gives raw decision values, a forest and k-NN give coarse probabilities. At 100 splits they would reach tens of millions of numbers, so they need a compact format such as Parquet, not CSV. |
| **Tune the decision threshold** on the training set of each split. | Not built. Macro-F1 and balanced accuracy use a threshold of 0.5. AUC does not depend on it. |
| **Random forest on the GPU.** | Unresolved. XGBoost ran out of memory at full size on the 4 GB GPU; a larger GPU is its next test. NVIDIA's cuML fit, but was 3 times slower than scikit-learn at 3,000 × 3,000, and not for lack of memory (section 6.2). |

### Reproducing the published labels

Planned, not done. Rebuild Brueffer et al.'s ER classifier from their
published gene weights. Apply it to GSE96058. Compare its output with the
stored `er prediction mgc` label. This would show how the ER label was
made, instead of inferring it. The steps are in
[Appendix C.1](appendix_c_data_details.md#c1-reproducing-the-er-label-planned-not-done).

## 10.2 Beyond this project

**A dataset where Higher Criticism should work.** HC is built for rare,
weak, independent signal. All three datasets have widespread, correlated
signal, so HC acted as a top-10% filter. A dataset with rare signal, or a
simulation with a known number of useful features, would test whether HC
picks its own count well. HC with the exact variance of the sorted
p-values needs no such test: it selects the same genes as HC2008
(section 4.4).

**An exact test for floor-heavy data.** Half the gene values sit at a
floor. Four numbers could describe each gene: the two class sizes and
each class's share at the floor. These give an exact null distribution
per gene. Such a test separates how often a gene is detected from how
high it is when detected. The U test merges both.

**Correlated genes as one unit.** Many methods pick different but
equivalent genes on each split. Selecting groups of genes that move
together could give stable, interpretable results.

**Stability-aware selection.** Stability selection (Meinshausen and
Bühlmann 2010) keeps the features chosen often across many subsamples.
It trades speed for stability. This study shows that stability is the
weak point of the best-scoring method, tree importance.

**Rebuilding the five-class prediction.** Each PAM50 subtype against the
rest is a separate binary task here. Combining the five classifiers into
one subtype prediction was set aside.

**A forest built for the GPU.** A forest could be grown level by level,
with all nodes of a level at once. Our analysis suggests this fits a
GPU's parallelism. No library we tried does this well on a 4 GB card.

**The frontier as a product.** A cost-quality frontier per dataset tells
a practitioner which selector fits their time budget. It could be
computed automatically for any `X, y`. The research harness would then be
useful beyond this study.
