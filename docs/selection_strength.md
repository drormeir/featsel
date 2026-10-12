# Measuring a selection method's strength against random selection

Discussion of 2026-10-12. Nothing here is built yet. The open decisions are
listed at the end.

## The problem

We want one number per selection method and dataset. It should say how much
better the method's genes are than genes chosen at random. The random selector
is the control: a method has to beat it to be worth anything.

## The flow

1. Split the patients into training and test patients.
2. The selector sees the training patients only. It sees every gene, and it
   returns a subset of the genes.
3. A classifier is trained on that subset, with the training patients only.
4. The classifier is scored on the test patients.
5. Repeat over many splits.

Each split may select a different subset. So the score judges the whole
procedure (select, tune, train), not one gene list or one model.

## Why the selector must not see the test patients

"Not see" means rows, not columns. The selector sees every gene, but none of
the test patients' rows.

If it sees all patients, it picks genes that happen to fit the test patients
by chance. The test then rewards that chance. On pure noise (100 patients,
5,000 random genes, random labels, 20 genes selected, 5-fold split):

| Where the 20 genes were chosen | Test AUC |
|---|---|
| On all patients | 0.96 |
| On the training patients only | 0.44 |
| The truth | 0.50 |

This is the selection bias described by Ambroise and McLachlan [1].

## The classifier

Logistic regression with a bias term and an L2 penalty, λ‖w‖².

- In scikit-learn, `C` plays the role of 1/λ.
- The `lbfgs` solver does not penalize the bias. `liblinear` does.
- Genes are standardized on the training patients first. Otherwise the
  penalty treats genes on different scales differently.
- λ is tuned by cross-validation inside the training patients. It maximizes
  the same score the evaluation uses. Its value is not reported.

Tuning inside the training patients is not leakage. The selector chose its
genes on those same patients, so the inner scores come out too optimistic.
That only affects which λ is picked. The final test score stays honest.

## The score

**Not F1.** F1 fixes a trade-off between false positives and false negatives.
That is a cost decision we do not want to make. Accuracy, balanced accuracy,
MCC and G-mean also depend on a cutoff, so each implies some costs too.

**AUC.** AUC asks how often the model scores a positive patient above a
negative one. It needs no cutoff and no costs.

AUC needs one model scoring a set of test patients. Comparing scores from
different models is not meaningful.

## Splitting: Monte Carlo or leave-one-out

**Monte Carlo.** Each split has one model and a set of test patients. So each
split gives a valid AUC. The median over splits summarizes the method, and
the spread between splits shows how certain the comparison is.

**Leave-one-out.** Each test set is one patient, so AUC cannot be computed per
round. Pooling all held-out scores into one AUC compares scores from n
different models. That is the problem above.

Leave-one-out can still give a confusion table. Each held-out patient gets a
decision from its own model, and counting decisions across models is
legitimate. The table describes the procedure as a whole. But it brings back
the cutoff, and with it the cost trade-off. It also gives one table with no
spread.

Leave-one-out trains on n − 1 patients, so its estimate is closest to the
full-data procedure. Monte Carlo trains on fewer patients and scores somewhat
pessimistically. Raising its training share, for example to 90%, narrows that
gap. Leave-one-out also costs n fits per setting: 3,069 for SCAN-B.

## A curve over the feature budget, and its area

A selector's sensitivity parameter imposes an order on the genes. Taking the
top k genes for k = 1, 2, ... gives one subset per k. Scoring each subset gives
a curve: score against k. The area under that curve summarizes the selector
over all budgets, with no single k to choose.

Subtracting the random selector's curve gives the gain over random selection,
over all budgets. That matches the project's headline question: which method
is best, and by how much over selecting blindly.

Four decisions make it well-defined:

1. **A score per k.** With Monte Carlo it can be AUC. A score taken from a
   confusion table brings back the cutoff.
2. **The k axis.** On a plain axis up to 9,259 genes, most of the area comes
   from large k, where every method scores alike. A log scale, or a cap such
   as k ≤ 1,000, keeps the area on the budgets where methods differ. This
   choice weights the budgets.
3. **The order is computed per training split.** It is never computed once on
   all patients.
4. **Not every selector gives an order.** The order is clean for nested
   selectors: a stricter setting keeps a subset of a looser one. ANOVA, FDR,
   FWE, Percentile, correlation and sequential selection are nested. So are
   mutual information and tree importance with a fixed seed. RFE has ties
   within each elimination round. Lasso by `C` and RFECV give no clean order.

In practice the curve is evaluated on a grid of k values, with λ tuned at each.

## Prior work

The area under a "performance against number of features" curve is published.
It is not a new method.

- FSDEM [2] defines a selection method's score as the area under M(f), any
  performance measure at f selected features, normalized by the range of f.
- Overschie [3] scores feature rankers by the mean validation score over
  feature counts. It notes this equals the area under the validation curve.
- Slavkov et al. [4] build one curve from the top-ranked features and one from
  the bottom-ranked. A good ranking does well on the first and badly on the
  second.
- Glaysher et al. [5] plot AUROC against the number of variables for several
  rankers, with random selection as a baseline.
- Tian et al. [6] add top-ranked features one at a time and compare
  selection methods on mammography data by AUC.

What this design adds is a combination of published parts: AUC per k, the
random selector subtracted, a weighted k axis, and the order computed inside
each training split.

## Open decisions

- The score per k: AUC with Monte Carlo, or a confusion-table score with
  leave-one-out.
- The k axis: log scale or a cap, and the grid of k values.
- How to handle selectors without a clean order.
- The Monte Carlo training share and number of splits.

## Limits

- One number hides where along k a method wins. The curve should be shown
  next to the area.
- Methods that choose their own count, such as FDR or Higher Criticism, give
  one point, not a curve. They need a separate comparison at their chosen k.

## References

[1] Ambroise, C. and McLachlan, G. J. (2002). Selection bias in gene
extraction on the basis of microarray gene-expression data. *PNAS* 99(10),
6562-6566.

[2] Rajabinasab, M., Lautrup, A. D., Hyrup, T. and Zimek, A. (2024). FSDEM:
Feature Selection Dynamic Evaluation Metric. SISAP 2024; extended version
arXiv:2408.14234.

[3] Overschie, J. G. S. (2021). A novel evaluation methodology for supervised
Feature Ranking algorithms. Technical report, University of Groningen.
arXiv:2207.04258.

[4] Slavkov, I., Petković, M., Geurts, P., Kocev, D. and Džeroski, S. (2020).
Error curves for evaluating the quality of feature rankings. *PeerJ Computer
Science* 6, e310. doi:10.7717/peerj-cs.310.

[5] Glaysher, P., Katzy, J. M. and An, S. (2019). Iterative subtraction method
for Feature Ranking. arXiv:1906.05718.

[6] Tian, C. J., Lv, J. and Xu, X. F. (2021). Evaluation of Feature Selection
Methods for Mammographic Breast Cancer Diagnosis in a Unified Framework.
*BioMed Research International* 2021, 6079163. doi:10.1155/2021/6079163.
