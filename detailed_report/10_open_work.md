# 10. Open work and next steps

[Back to contents](README.md) · Previous: [9. Conclusions](09_conclusions.md) · Next: [Appendix A: how to reproduce](appendix_a_reproduce.md)

Section 10.1 lists unfinished work within the project's four tasks.
Section 10.2 lists research beyond its scope.

## 10.1 Missing work within the project

### Experiments

| Item | Why it matters | Effort |
|---|---|---|
| **Tuned classifier.** C is fixed at 1 (section 3.8). | Absolute AUCs may be a little low. A method that needs a different C is at a disadvantage. | The config supports it. About 21 times the classifier fits: 4 values of C on 5 validation sets, plus a refit. Repeated inside each fold of the `k` search. |
| **The full SCAN-B grid**: every selector × classifier × `k` × train fraction. | The protocol uses one classifier and one train fraction. | Skipped by decision: too slow for this laptop. |
| **A measured target on SCAN-B**, such as the pathologist's ER status. | Every SCAN-B task here predicts a computed label. A measured label would test the conclusions against clinical truth. | One config. The loader supports it. |
| **The quantile transform with every classifier** (section 3.10). | Would show whether shrinkage LDA and k-NN gain from normal-shaped features. | One `k`, all classifiers. |
| **How SCAN-B computed `pam50 subtype`.** | The GEO notes say only that it implements Parker's method, after centring to a reference set, with unpublished R scripts (section 2.1). | Reading other SCAN-B papers. |
| **Reproduce the published ER classifier** and compare its output with the stored `er prediction mgc` label. | It would show how the ER label was made, instead of inferring it. | Planned, not done ([Appendix B.1](appendix_b_data_details.md#b1-reproducing-the-er-label-planned-not-done)). |

### Software

Config validation and other known limits are in section 5.8.

| Item | State |
|---|---|
| **Store each test patient's score**: the true label, the score and the decision threshold. Any metric could then be derived later without a new run. | Designed, not built. Scores differ across classifiers: a linear SVM gives raw decision values, a forest and k-NN give coarse probabilities. Each SCAN-B result holds 893 test patients, so 16 splits of the frontier would already store millions of numbers. They need a compact format such as Parquet, not CSV. |
| **Tune the decision threshold** on the training set of each split. | Not built. Macro-F1 and balanced accuracy use a threshold of 0.5. AUC does not depend on it. |
| **Random forest on the GPU.** | Unresolved. XGBoost ran out of memory at full size on the 4 GB GPU; a larger GPU is its next test. cuML was slower than scikit-learn (Appendix D.5). |
## 10.2 Beyond this project

**A dataset where Higher Criticism should work.** HC is built for rare,
weak, independent signal. All three datasets have widespread signal (on
SCAN-B also strongly correlated), so HC acted as a top-10% filter. A dataset with rare signal, or a
simulation with a known number of useful features, would test whether HC
picks its own count well.

**An exact test for floor-heavy data.** Half the gene values sit at a
floor. A test that separates how often a gene is detected from how high
it is when detected would fit them better than the U test, which merges
both (Appendix B.4).

**How often does a random set beat a chosen one?** Our random control
asks whether a method beats random features on average, at the same `k`.
A sharper question: for a gene set of size `k` chosen by a method, what
share of random size-`k` sets predict at least as well on the same split?
That share is an empirical p-value for the chosen set, and it can be
followed as `k` grows. A random set needs only a classifier fit, so
thousands of draws per `k` are affordable. A stronger version draws random
genes matched to the chosen ones in expression level and spread. In
practice, it would check whether a published gene signature predicts
better than chance signatures of its size.

**Correlated genes as one unit.** Many methods pick different but
equivalent genes on each split. Selecting groups of genes that move
together could give stable, interpretable results.

**Stability-aware selection.** Stability selection (Meinshausen and
Bühlmann 2010) keeps the features chosen often across many subsamples.
It trades speed for stability. This study shows that stability is the
weak point of tree importance (section 7.4).

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
