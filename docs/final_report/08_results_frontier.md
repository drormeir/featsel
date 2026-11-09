# 8. Results: cost against quality

[Back to contents](README.md) · Previous: [7. Results: quality](07_results_quality.md) · Next: [9. Conclusions](09_conclusions.md)

Chapter 7 asked which method selects best. A practitioner also asks what
each method costs. This chapter puts both on one plot.

## 8.1 What a cost-quality frontier is

Each point on the plot is one method at one `k`. Its x-position is the
selection time per fit (log scale), and its y-position is the test ROC
AUC. A point is **on the frontier** when no other point is both faster and
better. The frontier, drawn in blue, is the set of sensible choices: for
any time budget, the best method you can afford is on it. Points off the
frontier (gray) are dominated: something else is at least as good and
faster.

## 8.2 The experiment

| Setting | Value |
|---|---|
| Datasets | SCAN-B (five tasks) and BreastMNIST (one task) |
| Methods | Random; Mann-Whitney, HC2008, Spearman correlation, mutual information and Lasso (PPDNA) on the GPU; Lasso (liblinear), tree importance (200 trees, 4 threads), RFE (liblinear and Newton), RFECV (Newton) on the CPU; Sequential (Newton) on the GPU |
| `k` | 10, 50, 250, 1000 on SCAN-B; 10, 50, 250 on BreastMNIST; 5 and 10 for Sequential |
| Splits | 5 random splits, 70% of the samples for training |
| Classifier | logistic regression, C = 1 (not tuned, to keep the run short) |
| Score | ROC AUC; macro-F1 and balanced accuracy also recorded |
| Timing | median of 10 fits by fresh selectors (1 fit for Sequential); one fit at a time on the GPU |

The GPU methods ran with one worker, so no two fits shared the GPU. The
CPU methods ran with two workers. Every GPU method's fits were recorded as
running on the GPU.

Configs: [`configs/frontier_scanb70_*.yaml`](../../configs/),
[`configs/frontier_breastmnist70_*.yaml`](../../configs/). Plot script:
[`scripts/plot_frontier.py`](../../scripts/plot_frontier.py). Notebook with
all SCAN-B plots: [`notebooks/04_frontier_scanb70.ipynb`](../../notebooks/04_frontier_scanb70.ipynb).
The SCAN-B run took about 8 hours; BreastMNIST took about 6 minutes.

## 8.3 SCAN-B, all five tasks

![Frontier on SCAN-B](figures/frontier_scanb70_roc_auc.png)

*Figure 8.1. Selection time against test AUC, mean over 5 tasks and 5
splits. Bars: one standard deviation over splits. Table:
[`figures/frontier_scanb70_roc_auc.csv`](figures/frontier_scanb70_roc_auc.csv).*

| Method | Best AUC (its `k`) | Time per fit | Device | On frontier |
|---|---|---|---|---|
| Random | 0.950 (1000) | 0.02 s | CPU | yes, every `k` |
| HC2008 | **0.970** (2,855 genes) | 0.66 s | GPU | **yes** |
| Mann-Whitney | 0.966 (1000) | 0.66 s | GPU | no |
| Lasso, PPDNA | 0.962 (1000) | 0.73 s | GPU | no |
| Mutual information | 0.968 (1000) | 3.6 s | GPU | no |
| Spearman correlation | 0.966 (1000) | 3.6 s | GPU | no |
| Lasso, liblinear | 0.964 (1000) | 4.1 s | CPU | no |
| RFE, Newton | 0.959 (1000) | 6.0 s | CPU | no |
| Tree importance | **0.970** (1000) | 11.4 s | CPU | yes, by 0.00002 |
| RFECV, Newton | 0.739 (own count) | 31 s | CPU | no |
| RFE, liblinear | 0.959 (1000) | 57 s | CPU | no |
| Sequential, Newton | 0.939 (5) | 84 s | GPU | no |

**What it shows.**

1. **The cheap end is enough.** HC2008 and Mann-Whitney reach AUC 0.97 in
   0.66 s. Nothing slower is clearly better. Tree importance ties HC2008
   (0.970) but takes 17 times longer; split by split, each wins on some of
   the five splits.
2. **The GPU moved four methods into the top-left corner.** Mann-Whitney,
   HC2008, Lasso (PPDNA), mutual information and correlation all run in
   under 4 s, at AUC 0.96 to 0.97. Before the GPU work, Mann-Whitney took
   about 10 s and mutual information more than a minute per fit (chapter 6).
3. **The wrappers lose on both axes.** RFE, RFECV and Sequential take 6 to
   170 s and score below the simple filters. Sequential's 5 or 10 genes
   cannot match 1,000 well-chosen ones.
4. **Random genes come close.** Random k = 1000 reaches 0.950: once `k` is
   large, almost any genes work, as chapter 7 found.
5. **The top methods differ by about one standard deviation over splits**
   (0.003 to 0.016). With 5 splits a paired test cannot reach p < 0.06, so
   the order among the top methods is not established here. Chapter 7, with
   16 splits, is the place for that.

**Caution on HC2008.** It kept 2,855 genes, the edge of its search, on
every task and split, as in chapter 7. It is on the frontier because it is
as fast as Mann-Whitney and uses more genes. It is not a sign that HC
chose a better count.

## 8.4 SCAN-B, each task

| Task | Frontier beyond random selection (AUC, time) | Note |
|---|---|---|
| [Basal vs rest](figures/frontier_scanb70_roc_auc_ovr_Basal.png) | Lasso PPDNA k = 250 (0.992, 0.55 s), Mann-Whitney k = 10 (0.993, 0.67 s), mutual information k = 1000 (0.993, 3.7 s), Lasso liblinear k = 250 (0.993, 3.9 s) | The easiest task: almost every method scores 0.99, and random 1,000 genes 0.988. |
| [Her2 vs rest](figures/frontier_scanb70_roc_auc_ovr_Her2.png) | HC2008 (0.985, 0.66 s) | RFECV collapses to 0.63. |
| [LumA vs rest](figures/frontier_scanb70_roc_auc_ovr_LumA.png) | HC2008 (0.960, 0.65 s), tree importance k = 1000 (0.961, 11.4 s) | The only balanced task; RFECV works here (0.960). |
| [LumB vs rest](figures/frontier_scanb70_roc_auc_ovr_LumB.png) | HC2008 (0.970, 0.65 s), Mann-Whitney k = 1000 (0.974, 0.67 s) | RFECV collapses to 0.64. |
| [Normal vs rest](figures/frontier_scanb70_roc_auc_ovr_Normal.png) | Mann-Whitney k = 1000 (0.929, 0.65 s), HC2008 (0.947, 0.66 s) | The hardest task; RFECV at chance (0.50). |

Click a task to see its plot; each has a table beside it in
[`figures/`](figures/). On every task the frontier above random selection
holds only sub-second GPU filters, except tree importance on LumA and two
slower near-ties on Basal.

### Why RFECV failed on three tasks

RFECV chooses its own count by inner 5-fold cross-validation. In the
frontier it kept only 1 or 2 genes on Her2, LumB and Normal, giving AUC
0.50 to 0.64. On LumA it kept 8,567 to 25,697 genes and scored 0.96. On
Basal it kept 2,857 genes on four splits and 2 on one.

Our hypothesis: RFECV's default inner score is accuracy. On an imbalanced
task with a strong penalty (C = 0.01), the SVM predicts "rest" for almost
everyone at any gene count, so the accuracy curve is flat. A flat curve
makes the smallest count win the tie. LumA is the only balanced task, and
there RFECV works. Section 8.6 tests this hypothesis.

## 8.5 BreastMNIST

![Frontier on BreastMNIST](figures/frontier_breastmnist70_roc_auc.png)

*Figure 8.2. BreastMNIST, 5 splits. Table:
[`figures/frontier_breastmnist70_roc_auc.csv`](figures/frontier_breastmnist70_roc_auc.csv).*

| Method | Best AUC (its `k`) | Time per fit | On frontier |
|---|---|---|---|
| Random | 0.835 (250) | < 0.001 s | yes |
| Mann-Whitney | 0.867 (250) | 0.009 s | yes |
| HC2008 | 0.864 (204) | 0.009 s | no |
| Spearman correlation | 0.867 (250) | 0.02 s | no |
| Mutual information | 0.875 (250) | 0.07 s | yes |
| Lasso, liblinear | **0.902** (250) | 0.11 s | **yes** |
| Lasso, PPDNA | 0.901 (250) | 0.12 s | no |
| RFE, Newton | 0.900 (250) | 0.24 s | no |
| Tree importance | 0.888 (250) | 0.58 s | no |
| RFECV, Newton | 0.869 (own count) | 1.2 s | no |
| RFE, liblinear | 0.900 (250) | 1.7 s | no |
| Sequential, Newton | 0.788 (10) | 3.0 s | no |

- With 2,048 features and 546 training images, every method takes under
  3 s, so cost hardly matters here.
- **Lasso is the best choice**: 0.90 in 0.11 s. Here the methods that see
  features together (Lasso, RFE, trees) beat the one-at-a-time filters by
  about 0.03. Image embedding features are weakly correlated with each
  other, and a combination of them carries more than the sum of single
  features.
- RFECV works on this task (0.87).
- **Spearman correlation equals Mann-Whitney exactly** at every `k`. With a
  binary label, ranking features by |Spearman correlation| gives the same
  order as ranking by the U test's AUC: both depend only on the ranks of the
  feature within the two classes. On SCAN-B they differ slightly at large
  `k`, only because the correlation selector also drops one gene of each
  highly correlated pair.

## 8.6 RFECV: testing the hypothesis

*To be filled in when the check finishes.*

## 8.7 Summary

- On SCAN-B, sub-second GPU filters (Mann-Whitney, HC2008) give the best
  AUC for their cost; a 200-tree forest ties them at 17 times the cost.
- On BreastMNIST, Lasso is best, and every method is cheap.
- Wrappers cost the most and gave the least on both datasets.
- RFECV with its default inner score fails on imbalanced tasks.
