# 10. Open work and next steps

[Back to contents](README.md) · Previous: [9. Conclusions](09_conclusions.md) · Next: [Appendix A: glossary](appendix_a_glossary.md)

This chapter lists what is missing. Section 10.1 is work that
belongs to this project's four tasks but is not finished. Section 10.2 is
research beyond the project's scope.

## 10.1 Missing work within the project

### Experiments

| Item | Why it matters | Effort |
|---|---|---|
| **More splits for the frontier.** It has 5 splits. The comparison has 16. | With 5 splits the corrected paired test has 4 degrees of freedom. So the order among the top methods on the frontier is not established. | About 8 h per 5 extra splits on SCAN-B. |
| **Tuned classifier in the frontier.** C is fixed at 1 there. | Absolute AUCs may be a little low. A method needing a different C is disadvantaged. | Multiplies the classifier's cost by about 15. |
| **A paired test of selection against all features.** Section 8.7 found gains of 0.006 AUC on SCAN-B and a loss on BreastMNIST, over 5 splits. | Whether selection improves the classifier at all is a central question. 5 splits cannot settle a gain of this size. | Add the all-features baseline to the 16-split comparison. Hours. |
| **RFECV scored by AUC in the 16-split comparison.** On the frontier (5 splits, section 8.6) it no longer collapsed on SCAN-B. | Would give it the paired tests of chapter 7. | About 6 hours on SCAN-B. |
| **RFE with a finer step and a forest with more trees in the comparison.** The comparison used RFE removing half the genes per round and 50 trees. | Both were chosen for speed. The frontier used step 0.1 and 200 trees, and RFE stayed weakest. So the conclusion is unlikely to change. | Hours. |
| **The full SCAN-B grid**: every selector × classifier × `k` × train fraction. | The comparison uses one classifier and one train fraction. | About four days on this laptop. Skipped by decision. |
| **A measured target on SCAN-B**, such as the pathologist's ER status. | Every SCAN-B task here predicts a computed label. A measured label would show whether the conclusions hold for clinical truth. | One config. The loader already supports it. |
| **The quantile transform with every classifier** (section 3.10). With logistic regression it did not help. | Would show whether shrinkage LDA and k-NN gain from normal-shaped features. | Started and cancelled. One `k`, all classifiers. |
| **Time and memory complexity per method**, from the algorithm. | Planned to sit next to the measured times. A disagreement would be a finding. | Writing, no runs. |
| **How SCAN-B computed `pam50 subtype`.** | The GEO notes say only that it implements Parker's method, after centring to a reference set, with unpublished R scripts (section 2.1, Appendix C.2). The exact implementation is unknown. | Reading other SCAN-B papers. |

### Software

| Item | State |
|---|---|
| **Store each test patient's score**: the true label, the model's score and the decision threshold, so every metric can be derived later and a new metric needs no new run. | Designed, not built. Scores are not comparable across classifiers. A linear SVM gives raw decision values. A forest and k-NN give coarse probabilities. At 100 splits the scores would run to tens of millions of numbers. So they need a compact format such as Parquet, not CSV. |
| **Tune the decision threshold** inside the split. | Not built. Macro-F1 and balanced accuracy use 0.5. AUC does not depend on it. |
| **Random forest on the GPU.** | XGBoost ran out of memory at full size on the 4 GB GPU. NVIDIA's cuML fit, but was 3 times slower than scikit-learn at 3,000 × 3,000, and not for lack of memory (section 6.2). Unresolved. A larger GPU is the next test for XGBoost. |
| **The speed notebooks' overlap column** past the original's largest size. | A bug in the benchmark's reference handling. Where the original stopped growing, the overlap is measured against the first variant that ran, not the original. This affects RFECV, tree importance and mutual information at 30,000 genes, and Sequential at 3,000 and 10,000 genes. |
| **Validate a config before the run starts.** | A typo fails only when the first cell that uses it runs. |

### Reproducing the published labels

Planned, not done. Rebuild Brueffer et al.'s ER classifier
from their published gene weights. Apply it to GSE96058. Then compare it
with the stored `er prediction mgc` label. This would show exactly how the ER
label was made, instead of inferring it. The plan, step by step, is in
[Appendix C.1](appendix_c_data_details.md#c1-reproducing-the-er-label-planned-not-done).

## 10.2 Beyond this project

**A dataset where Higher Criticism should work.** HC is built for rare,
weak, independent signal. All three of our datasets have widespread,
correlated signal, so HC acted as a top-10% filter. A dataset with rare
signal, or a simulation with a known number of useful features, would test
whether HC picks its own count well. HC with the exact variance of the
sorted p-values needs no test of its own: it selects the same genes as
HC2008 (section 4.4).

**An exact test for floor-heavy data.** Each gene could be described by
four numbers. Two are the class sizes. The other two are the share of
each class at the floor. That gives an exact null distribution per gene.
It separates "how often is the gene detected" from "how high is it when
detected". The U test merges both.

**Correlated genes as one unit.** Many methods pick different but
equivalent genes on each split. Grouping genes that move together, and
selecting groups, could give stable and interpretable results.

**Stability-aware selection.** Stability selection (Meinshausen and
Bühlmann 2010) keeps features chosen often across many subsamples. It
trades speed for stability. This study shows that stability is the weak
point of the best-scoring method.

**Rebuilding the five-class prediction.** The project selects features
for binary tasks. Combining the five subtype-vs-rest classifiers into one
subtype prediction was parked.

**A forest built for the GPU.** A forest could be grown level by level,
with all nodes of a level at once. Our analysis suggests this would fit a
GPU's parallelism. No library we tried does this well on a 4 GB card.

**The frontier as a product.** A cost-quality frontier per dataset tells a
practitioner which selector to use for their time budget. It could be
automated for any dataset given as `X, y`. That would make the package's
research harness useful beyond this study.
