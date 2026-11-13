# 10. Open work and next steps

[Back to contents](README.md) · Previous: [9. Conclusions](09_conclusions.md) · Next: [Appendix A: glossary](appendix_a_glossary.md)

This chapter lists what is missing. Section 10.1 is work that belongs to
the project's four tasks but is not finished. Section 10.2 is research
beyond the project's scope.

## 10.1 Missing work within the project

### Experiments

| Item | Why it matters | Effort |
|---|---|---|
| **Tuned classifier in the frontier.** The classifier's C is fixed at 1 there. | Absolute AUCs may be a little low. A method that needs a different C is at a disadvantage. | Multiplies the classifier's cost by about 15. |
| **RFECV scored by AUC in the 16-split comparison.** On the frontier (5 splits, section 8.6) it no longer collapsed on SCAN-B. | Would give it the paired tests of chapter 7. | About 6 hours on SCAN-B. |
| **RFE with a finer step and a forest with more trees in the comparison.** In the comparison, RFE removed half the genes per round, and the forest had 50 trees. | Both settings were chosen for speed. The frontier used step 0.1 and 200 trees, and RFE stayed weakest. So the conclusion is unlikely to change. | Hours. |
| **The full SCAN-B grid**: every selector × classifier × `k` × train fraction. | The comparison uses one classifier and one train fraction. | About four days on this laptop. Skipped by decision. |
| **A measured target on SCAN-B**, such as the pathologist's ER status. | Every SCAN-B task here predicts a computed label. A measured label would show whether the conclusions hold for clinical truth. | One config. The loader already supports it. |
| **The quantile transform with every classifier** (section 3.10). With logistic regression it did not help. | Would show whether shrinkage LDA and k-NN gain from normal-shaped features. | One `k`, all classifiers. A run was started and cancelled. |
| **Time and memory complexity per method**, derived from the algorithm. | Planned to sit next to the measured times. If the two disagree, that is a finding. | Writing only, no runs. |
| **How SCAN-B computed `pam50 subtype`.** | The GEO notes say only that it implements Parker's method, after centring to a reference set, with unpublished R scripts (section 2.1, Appendix C.2). The exact implementation is unknown. | Reading other SCAN-B papers. |

### Software

| Item | State |
|---|---|
| **Store each test patient's score**: the true label, the model's score and the decision threshold. Every metric could then be derived later. A new metric would need no new run. | Designed, not built. Scores are not comparable across classifiers. A linear SVM gives raw decision values. A forest and k-NN give coarse probabilities. At 100 splits the scores would reach tens of millions of numbers. So they need a compact format such as Parquet, not CSV. |
| **Tune the decision threshold** on the training set of each split. | Not built. Macro-F1 and balanced accuracy use a threshold of 0.5. AUC does not depend on it. |
| **Random forest on the GPU.** | XGBoost ran out of memory at full size on the 4 GB GPU. NVIDIA's cuML fit, but was 3 times slower than scikit-learn at 3,000 × 3,000, and not for lack of memory (section 6.2). Unresolved. A larger GPU is the next test for XGBoost. |
| **Validate a config before the run starts.** | Not built. A typo fails only when the first cell that uses it runs. |

### Reproducing the published labels

Planned, not done. Rebuild Brueffer et al.'s ER classifier from their
published gene weights. Apply it to GSE96058. Compare its output with the
stored `er prediction mgc` label. This would show exactly how the ER label
was made, instead of inferring it. The plan, step by step, is in
[Appendix C.1](appendix_c_data_details.md#c1-reproducing-the-er-label-planned-not-done).

## 10.2 Beyond this project

**A dataset where Higher Criticism should work.** HC is built for rare,
weak, independent signal. All three of our datasets have widespread,
correlated signal, so HC acted as a top-10% filter. A dataset with rare
signal would test whether HC picks its own count well. So would a
simulation with a known number of useful features. HC with the exact
variance of the sorted p-values needs no such test. It selects the same
genes as HC2008 (section 4.4).

**An exact test for floor-heavy data.** Half the gene values sit at a
floor. Each gene could be described by four numbers. Two are the class
sizes. The other two are the share of each class at the floor. These give
an exact null distribution per gene. Such a test separates two questions:
how often the gene is detected, and how high it is when detected. The
U test merges both.

**Correlated genes as one unit.** Many methods pick different but
equivalent genes on each split. Grouping genes that move together, and
selecting groups, could give stable and interpretable results.

**Stability-aware selection.** Stability selection (Meinshausen and
Bühlmann 2010) keeps the features chosen often across many subsamples. It
trades speed for stability. This study shows that stability is the weak
point of the best-scoring method, tree importance.

**Rebuilding the five-class prediction.** The project treats each PAM50
subtype against the rest as a separate binary task. Combining the five
classifiers into one subtype prediction was set aside.

**A forest built for the GPU.** A forest could be grown level by level,
with all nodes of a level at once. Our analysis suggests this would fit a
GPU's parallelism. No library we tried does this well on a 4 GB card.

**The frontier as a product.** A cost-quality frontier per dataset tells a
practitioner which selector fits their time budget. It could be computed
automatically for any dataset given as `X, y`. The research harness would
then be useful beyond this study.
