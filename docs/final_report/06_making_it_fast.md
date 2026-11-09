# 6. Making it fast (task 3)

[Back to contents](README.md) · Previous: [5. The software](05_the_software.md) · Next: [7. Results: quality](07_results_quality.md)

Task 3 asks for infrastructure that speeds up feature selection. We
attacked it at two levels: the experiment as a whole (the runner), and each
method on its own (faster algorithms, and a GPU).

**A rule we kept:** a speed claim comes only from a series of measurements
on several sizes, never from one timing. Every number in this chapter comes
from such a series, run on one laptop (Intel i7-9750H, 12 threads, 16 GB
RAM; NVIDIA GTX 1650, 4 GB).

## 6.1 The runner against plain scikit-learn

The same small grid ran four ways: scikit-learn's `cross_validate`, and
our runner, each on one core and on all 12 threads. The script checks that
all four give the same test scores before it reports a time.

SCAN-B, Basal vs rest, 3,000 random genes, 2,976 patients, 5 splits, Lasso
and RFE at `k` = 10, 50 and 250:

| Run | Seconds | Speed-up over scikit-learn on 1 core |
|---|---|---|
| scikit-learn, 1 core | 33.6 | 1.0 |
| runner, 1 core | 10.4 | 3.2 |
| scikit-learn, 12 threads | 22.3 | 1.5 |
| runner, 12 threads | 6.3 | 5.3 |

**Why.** The runner does less work, not different work:

- It prepares each split (imputation, scaling) once and shares it with
  every task, selector and classifier.
- It fits each rank-once selector once per split and cuts the ranking at
  every `k` (chapter 5). scikit-learn refits it for every `k`.
- It runs jobs in parallel, with the data in shared memory.

On this small grid, starting the worker processes took about 4.5 of the
6.3 seconds, so parallel work added little. On the full data each fit is
longer and start-up weighs less; that case was not measured separately.

Source: [`docs/speed_benchmark.md`](../speed_benchmark.md),
[`results/speed_benchmark.csv`](../../results/speed_benchmark.csv).

## 6.2 Each method, faster

Each selector has a speed notebook,
`featsel/selectors/<name>/demos/speed.ipynb`. It times the scikit-learn
original and our versions on synthetic data shaped like SCAN-B: 3,000
samples, 1,000 to 30,000 features, binary and imbalanced (80/20). It also
checks that every fit is reproducible and picks the same features as the
original. Each time is the median of the timed fits, after one warm-up
fit. A variant whose fit takes more than a minute stops growing.

Results at the largest size both sides reached:

| Method | Original | Ours, CPU | Ours, GPU | Same features? | Notebook |
|---|---|---|---|---|---|
| ANOVA, FDR, FWE, Percentile (F-test) | 0.36 s | 0.36 s | 0.14 s (2.6×) | yes | [anova_f](../../featsel/selectors/anova_f/demos/speed.ipynb) |
| Variance threshold | 0.83 s | 0.86 s | 0.13 s (6.4×) | yes | [variance_threshold](../../featsel/selectors/variance_threshold/demos/speed.ipynb) |
| Correlation (pandas full matrix) | 7.05 s | 6.89 s | 0.25 s (28×) | yes | [correlation](../../featsel/selectors/correlation/demos/speed.ipynb) |
| Mann-Whitney and Higher Criticism (our NumPy U test) | 3.76 s | — | 0.49 s (7.7×) | yes | [mann_whitney](../../featsel/selectors/mann_whitney/demos/speed.ipynb) |
| Mutual information, 10,000 features | 78.8 s | 18.4 s (4.3×) | 1.60 s (49×) | yes, bit for bit | [mutual_info](../../featsel/selectors/mutual_info/demos/speed.ipynb) |
| Lasso (liblinear → PPDNA solver) | 5.66 s | — | 1.08 s (5.2×) | 91% overlap (see below) | [lasso](../../featsel/selectors/lasso/demos/speed.ipynb) |
| Tree importance, 10,000 features (1 thread → 12) | 103.3 s | 18.9 s (5.5×) | no GPU path | yes | [tree_importance](../../featsel/selectors/tree_importance/demos/speed.ipynb) |
| RFE (liblinear → exact Newton solver) | 26.1 s | 12.7 s (2.1×) | 15.4 s (1.7×) | not at every size (see below) | [rfe](../../featsel/selectors/rfe/demos/speed.ipynb) |
| RFECV, 10,000 features | 76.6 s | 51.8 s (1.5×) | 44.4 s (1.7×) | not at every size | [rfecv](../../featsel/selectors/rfecv/demos/speed.ipynb) |
| Sequential forward, 1,000 features | 191.3 s | 22.5 s (8.5×) | 4.9 s (39×) | yes | [sequential](../../featsel/selectors/sequential/demos/speed.ipynb) |
| Random | — | 0.02 s | — | — | [random_selection](../../featsel/selectors/random_selection/demos/speed.ipynb) |

Sizes are 30,000 features unless the row says otherwise. Every fit of
every variant was reproducible.

On real SCAN-B (2,976 patients, 28,552 genes), the Lasso notebook also
times the five tasks: PPDNA on the GPU took 0.7 to 1.4 s per fit against
4.5 to 5.8 s for liblinear, 3.2 to 7.0 times faster.

### Where the speed comes from

| Method | What changed |
|---|---|
| U test | One sort per gene gives the ranks and the tie groups at once; on the GPU, PyTorch sorts thousands of genes in parallel. Only the rank sums and tie counts are computed on the GPU, exactly, in 64-bit floats; the p-value is finished on the CPU, so every device gives identical results. |
| F-test, variance | Class sums as one vector product per class, on the GPU. |
| Correlation | The gene-gene correlations are computed in blocks, so the full 28,552 × 28,552 matrix is never held in memory. |
| Mutual information | scikit-learn's estimate (Ross 2014) searches nearest neighbours. For one feature at a time, that search is a sort. Our version sorts, and reproduces scikit-learn's scores bit for bit, including its random noise and its order of summation. |
| Lasso | PPDNA (Liu and Zhou 2023), a semismooth Newton method, solves the same L1 problem to a tight tolerance on the GPU. |
| RFE, RFECV | An exact Newton solver for the linear SVM (Chapelle 2007), working in the space of the samples (3,000) rather than the genes (30,000), and reusing work between rounds. |
| Sequential | All candidate features of one step are solved at once, as one batched Newton problem on the GPU, starting from the previous step's solution. |
| Tree importance | scikit-learn's own forest, on all threads. |

### Why some methods do not pick the same features

**Lasso.** liblinear stops at a tolerance of 10⁻⁴, far from the exact
solution. On Basal vs rest it keeps 330 genes where the exact solution has
196. PPDNA reaches the solution, so it picks fewer, different genes. Both
are reproducible. We keep liblinear as the default, since it matches
scikit-learn, and PPDNA as an option.

**RFE.** The same cause: liblinear's tolerance changes which genes are
removed when genes are strongly correlated. On the synthetic data at
3,000 features, the overlap (Jaccard index) between the two gene sets was
only 0.28. Again, liblinear stays the default.

### Why the GPU does not always win

- **64-bit arithmetic.** The GTX 1650, a consumer card, runs 64-bit
  floating point at 1/32 of its 32-bit speed. RFE's solver needs 64 bits,
  since 32 bits changed which genes it selected. So RFE is slower on this
  GPU than on the CPU.
- **Copying.** Data must be copied to the GPU. For the fast filters the
  copy takes about half the time.
- **Memory.** 4 GB is small. A random forest on the GPU (XGBoost) ran out
  of memory at full size; NVIDIA's cuML forest fit, but was 3 times slower
  than scikit-learn on 3,000 × 3,000. Tree importance therefore stays on
  the CPU.

## 6.3 The device policy

Nine selectors share one rule for choosing where to run:

| `device` | Behaviour |
|---|---|
| `'auto'` (default) | Use the GPU if one is free now; otherwise the CPU. |
| `'cpu'` | Always the CPU. |
| `'cuda'` | Wait for the GPU and use it. |
| any | A tensor already on a GPU runs there. |

`'auto'` never waits, so a parallel run never stalls on the GPU. The
frontier runs (chapter 8) set `'cuda'`, so every timing of a GPU method is
really on the GPU. The device used is recorded with every result.

## 6.4 Summary

- The runner is 3.2 times faster than plain scikit-learn on one core and
  5.3 times on twelve threads, with identical scores.
- On the GPU, the filters run 2.6 to 49 times faster and Sequential
  selection 39 times faster, with the same features.
- Lasso and RFE gained exact solvers that are faster but, because they
  are exact, pick somewhat different genes than scikit-learn's default.
- A cheap 4 GB GPU limits what can move to it: 64-bit work and large
  forests do not pay.
