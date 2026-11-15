# Appendix D. Engineering notes

[Back to contents](README.md)

Chapter 6 gives the speed results and the reasons behind them. This
appendix keeps the detail behind them: how the timings were checked, how
the Lasso solver was made to work, and what the GPU trials found. Like
chapter 6, its SCAN-B measurements use 28,552 genes, before the 11 flat
genes were dropped (section 2.1).

## D.1 How the timings were checked

The mutual information, tree importance, RFECV and Sequential notebooks
were run again after a fix. Past the original's largest size, the overlap
check had compared two of our variants instead of one with the original.
That overlap is now left empty. Most times moved by 7% or less.
scikit-learn's Sequential selection took 164 s, against 191 s before, so
the GPU's lead fell from 39 to 34 times. Two timed fits cannot pin a time
closer than that.

The RFE, RFECV and Lasso results were rerun on 2026-11-14, after
liblinear was dropped, against liblinear at its defaults. They ran as a
plain script,
[`scripts/benchmark_speed_selectors.py`](../scripts/benchmark_speed_selectors.py),
with nothing else running. Inside Jupyter, the same call ran Newton
slower on the CPU: 61 s against 41 s for RFECV at 3,000 features. The
cause was not found. It was not heat: ten fits in a row held their time,
clock and temperature. The script's RFECV times match the notebooks'
first run within 2%.

## D.2 Lasso on real SCAN-B

The script also times Lasso on the five tasks of real SCAN-B (2,976
patients, 28,552 genes):

| One Lasso fit per task | Seconds | Against liblinear |
|---|---|---|
| liblinear | 4.8 to 5.7 | — |
| PPDNA, GPU | 0.71 to 1.36 | 3.9 to 7.0 times faster |
| PPDNA, CPU | 4.5 to 31.3 | faster on Basal and Normal, up to 5.9 times slower on LumB |

The notebook standardizes the data in 32-bit floats, as the runner holds
it. A few genes are nearly constant: their spread is 10⁻⁶ to 10⁻⁹ around
the expression floor. 32-bit floats cannot hold so small a spread.
Standardizing them gives values up to 0.06 apart from the 64-bit result.
Only 0.02% of the values move by more than 0.001. PPDNA on the CPU is
sensitive to this. On LumB it took 35 s on data standardized in 32 bits
and 7.3 s in 64 bits, and kept 730 genes against 735. This is why the 11
flat genes were later dropped (section 2.1).

## D.3 Why PPDNA, and how ours departs from the paper

Simpler solvers failed. FISTA, even on a working set of genes, only tied
liblinear. Parallel coordinate descent overshot on correlated genes.
SSNAL handles only the squared loss. Our PPDNA departs from the paper
(Liu and Zhou 2023) in three ways, each needed on SCAN-B-shaped data:

- Its proximal step size starts at 40 C and grows five-fold per outer
  step, up to 10⁶ C. Beyond that, rounding loses the digits the weights
  are made of.
- The Newton matrix is factored in 32-bit floats, where this GPU is fast.
  The solution is refined in 64 bits.
- It also stops when three outer steps fail to halve the best residual.
  On 28,552 genes, rounding sets a floor near 10⁻⁹, above the target of
  10⁻¹⁰. Without this rule, Basal ran all 1,000 outer steps in 43 s.
  With it, 0.73 s (single timings), and every task stopped within 7 to
  10 outer steps.

## D.4 Lasso: speed against liblinear

For Lasso the speed argument for dropping liblinear is weaker than for
RFE. On the CPU, with C chosen from four values by 5-fold validation AUC,
both solvers took the same total time over the five SCAN-B tasks: 379 s
for liblinear, 381 s for PPDNA (single fits, on data standardized in 64
bits). PPDNA was faster on Basal, Her2 and Normal and slower on LumA and
LumB. Its cost grows with the number of genes the model keeps, and the
luminal tasks keep the most. Both chose C = 0.1 on every task. Their
validation AUCs agreed to the fourth digit. Only on a GPU, which
liblinear cannot use, was PPDNA clearly faster (section D.2).

Unlike every other wrapped selector, Lasso has no test that it is at most
5% slower than scikit-learn. No scikit-learn solver reaches Lasso's exact
solution. Its Newton solvers refuse the L1 penalty. liblinear and saga
accept it but stop at a tolerance: less work, a different answer. On
small synthetic data, PPDNA on the CPU was 3 to 4 times slower than
liblinear at its defaults. So Lasso's speed comparison is the SCAN-B one
of section D.2.

## D.5 The random forest on a GPU

A 4 GB GPU is small. The forest trials:

| Forest | Data (samples × features) | Result |
|---|---|---|
| XGBoost, GPU | 1,488 × 28,552 (synthetic, SCAN-B-shaped) | out of memory even with 64 bins, needing 3.4 GB |
| XGBoost, CPU | 1,488 × 28,552 | 339 s per fit |
| scikit-learn, 12 threads | 1,488 × 28,552 | 4.2 s per fit |
| XGBoost, GPU, depth 6 | 3,000 × 3,000 | 6.5 s, a tie with scikit-learn's 6.7 s |
| XGBoost, GPU, depth 6 | 3,000 × 30,000 | out of memory twice, asking for 3.5 GB in one block |
| cuML, GPU | 3,000 × 3,000 | 21.5 s, 3 times slower than scikit-learn's 6.7 s |

Only 21 of XGBoost's top 50 genes were scikit-learn's. That does not show
a different method. scikit-learn under another seed shares only about 20
of its own top 50 at full size (Jaccard 0.25). cuML's slowness was not
memory. Capping its trees at depth 8 took 4.5 s, at depth 16 15.1 s, and
with no cap 21.1 s. So most of the time goes to the middle levels, where
nodes are many. Tree importance therefore stays on the CPU.

## D.6 Libraries we did not use

JAX was rejected for the U test. It is a new dependency, reserves most
GPU memory per process, and recompiles for every data shape. CuPy and
Numba were not needed. Library sorts in PyTorch were enough, and PyTorch
is already needed for the embeddings.
