# 10. Open work and next steps

[Back to contents](README.md) · Previous: [9. Conclusions](09_conclusions.md) · Next: [Appendix A: decisions](appendix_a_decisions.md)

This chapter lists what is missing, honestly. Section 10.1 is work that
belongs to this project's four tasks and is not finished. Section 10.2 is
research beyond the project's scope. Section 10.3 is the dated list of
parked ideas.

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
| **Remove BreastMNIST's copied images and rerun.** 155 of 780 source images are copies (Pawłowska et al. 2023), some labelled differently. | Every BreastMNIST result carries this leak. The letter's supplement lists the pairs, so the copies can be dropped or kept on one side of each split. | A loader option and two reruns: minutes. |
| **The quantile-transform side experiment** (section 3.10). | Would show whether shrinkage LDA and k-NN gain from normal-shaped features. | Started and cancelled; one `k`, all classifiers. |
| **Time and memory complexity per method**, from the algorithm. | Planned to sit next to the measured times; a disagreement would be a finding. | Writing, no runs. |
| **How SCAN-B computed `pam50 subtype`.** | Neither Brueffer et al. nor its supplement says (Appendix D.2). | Reading other SCAN-B papers. |

### Software

| Item | State |
|---|---|
| **Store each test patient's score**: the true label, the model's score and the decision threshold, so every metric can be derived later and a new metric needs no new run. | Designed, not built. Scores are not comparable across classifiers: a linear SVM gives raw decision values, a forest and k-NN coarse probabilities. At 100 splits they would run to tens of millions of numbers, so they need a compact format such as Parquet, not CSV. |
| **Tune the decision threshold** inside the split. | Not built; macro-F1 and balanced accuracy use 0.5. AUC does not depend on it. |
| **Random forest on the GPU.** | XGBoost ran out of memory on the 4 GB GPU; NVIDIA's cuML was 3 times slower than scikit-learn. Unresolved; a larger GPU is the next test. |
| **The RFECV speed notebook's overlap column** at 30,000 genes. | A bug in the benchmark's reference handling makes that one overlap value meaningless. |
| **Validate a config before the run starts.** | A typo fails only when the first cell that uses it runs. |

### Reproducing the published labels

Decided 2026-10-17 and not done: rebuild Brueffer et al.'s ER classifier
from their published gene weights, apply it to GSE96058, and compare with
the stored `er prediction mgc` label. This would show exactly how the ER
label was made, instead of inferring it. The plan, step by step, is in
[Appendix D.1](appendix_d_data_details.md#d1-reproducing-the-er-label-planned-not-done).

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

## 10.3 Parked ideas

Ideas that are interesting but serve none of the four tasks in
`SCOPE.md`. Writing an idea down here is how it stops competing for
attention. Format: the date, the idea, and why it was parked.

- 2026-07-30 - A PyTorch layer and trainer for transfer learning on tiny
  image sets (ResNet embeddings, 2 classes × 10 samples). Parked: a second
  project with its own failure modes, and logistic regression on selected
  embeddings is the stronger baseline at this sample size.
- 2026-07-30 - A PyPI release of `featsel`. Parked: packaging polish is
  not a project requirement.
- 2026-10-05 - Higher Criticism with exact null moments: centre the i-th
  p-value at i/(N+1) and scale by √(N+2), from Beta(i, N−i+1), with a
  proof. Parked: no publication found, and `SCOPE.md` allows published HC
  only. (HC2008 here already uses the exact centre, i/(N+1); section 4.4.)
- 2026-10-05 - Berk-Jones (Donoho and Jin 2004, eq. 1.9) as a threshold
  rule, another Higher Criticism mode. Parked: published only as a
  detection test, not for feature selection.
- 2026-10-18 - Recombine the five PAM50 class-vs-rest classifiers into a
  decision tree that predicts the subtype. Parked: the project selects
  features for binary targets only.
