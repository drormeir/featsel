# 10. Open work and next steps

[Back to contents](README.md) · Previous: [9. Conclusions](09_conclusions.md) · Next: [Appendix A: decisions](appendix_a_decisions.md)

This chapter lists what is missing, honestly. Section 10.1 is work that
belongs to this project's four tasks and is not finished. Section 10.2 is
research beyond the project's scope.

## 10.1 Missing work within the project

### Experiments

| Item | Why it matters | Effort |
|---|---|---|
| **More splits for the frontier.** It has 5 splits; the comparison has 16. | With 5 splits no paired test can reach p < 0.06, so the order among the top methods on the frontier is not established. | About 8 h per 5 extra splits on SCAN-B. |
| **Tuned classifier in the frontier.** C is fixed at 1 there. | Absolute AUCs may be a little low, and a method needing a different C is disadvantaged. | Multiplies the classifier's cost by about 15. |
| **A paired test of selection against all features.** Section 8.7 found gains of 0.005 AUC on SCAN-B and a loss on BreastMNIST, over 5 splits. | Whether selection improves the classifier at all is a central question; 5 splits cannot settle a gain of this size. | Add the all-features baseline to the 16-split comparison; hours. |
| **RFECV scored by AUC in the 16-split comparison.** The frontier rerun (5 splits) fixed its collapse. | Would give it the paired tests of chapter 7. | About 6 hours on SCAN-B. |
| **RFE with a finer step and a forest with more trees in the comparison.** The comparison used RFE removing half the genes per round and 50 trees. | Both were chosen for speed. The frontier used step 0.1 and 200 trees, and RFE stayed weakest, so the conclusion is unlikely to change. | Hours. |
| **The full SCAN-B grid**: every selector × classifier × `k` × train fraction. | The comparison uses one classifier and one train fraction. | About four days on this laptop; skipped by decision. |
| **A measured target on SCAN-B**, such as the pathologist's ER status. | Every SCAN-B task here predicts a computed label. A measured label would show whether the conclusions hold for clinical truth. | One config; the loader already supports it. |
| **The size of the repeat-sample leak.** 93 patients appear twice in the course's copy. | Says how much the course's earlier results are inflated. | A paired run with and without the repeats. |

### Software

| Item | State |
|---|---|
| **Store each test patient's score** (`docs/experiment_design.md` 2c), so new metrics need no new run. | Designed, not built. |
| **Tune the decision threshold** inside the split. | Not built; macro-F1 and balanced accuracy use 0.5. AUC does not depend on it. |
| **Random forest on the GPU.** | XGBoost ran out of memory on the 4 GB GPU; NVIDIA's cuML was 3 times slower than scikit-learn. Unresolved; a larger GPU is the next test. |
| **The RFECV speed notebook's overlap column** at 30,000 genes. | A bug in the benchmark's reference handling makes that one overlap value meaningless. |
| **Validate a config before the run starts.** | A typo fails only when the first cell that uses it runs. |

### Reproducing the published labels

Decided 2026-10-17 and not done: rebuild Brueffer et al.'s ER classifier
from their published gene weights, apply it to GSE96058, and compare with
the stored `er prediction mgc` label. This would show exactly how the ER
label was made, instead of inferring it.

## 10.2 Beyond this project

**A dataset where Higher Criticism should work.** HC is built for rare,
weak, independent signal. Both our datasets have widespread, correlated
signal, so HC acted as a top-10% filter. A dataset with rare signal, or a
simulation with a known number of useful features, would test whether HC
picks its own count well. Two variants were parked because they are not
published as selection rules: HC with the exact variance of the sorted
p-values (scaling by N + 2), and the Berk-Jones statistic.

**An exact test for floor-heavy data.** Each gene could be described by
four numbers: the two class sizes and the share of each class at the
floor. That gives an exact null distribution per gene, separating "how
often is the gene detected" from "how high is it when detected". The U
test merges both.

**Correlated genes as one unit.** Many methods pick different but
equivalent genes on each split. Grouping genes that move together, and
selecting groups, could give stable and interpretable results.

**Stability-aware selection.** Stability selection (Meinshausen and
Bühlmann 2010) keeps features chosen often across many subsamples. It
trades speed for stability, which this study shows is the weak point of
the best-scoring method.

**Rebuilding the five-class prediction.** The project selects features
for binary tasks. Combining the five subtype-vs-rest classifiers into one
subtype prediction was parked.

**A forest built for the GPU.** Our analysis suggests a forest grown level
by level, all nodes of a level at once, would fit a GPU's parallelism. No
library we tried does this well on a 4 GB card.

**The frontier as a product.** A cost-quality frontier per dataset tells a
practitioner which selector to use for their time budget. Automating it,
for any dataset given as `X, y`, would make the package's research harness
useful beyond this study.
