# Appendix D. Engineering notes

[Back to contents](README.md)

Chapter 6 gives the speed results and the reasons behind them. This
appendix keeps the detail behind them: how the timings were checked, how
the Lasso solver was made to work, and what the GPU trials found. Like
chapter 6, its SCAN-B measurements use 28,552 genes, the 11 flat genes
included (section 2.1).

## D.1 How the timings were checked

Past the largest size scikit-learn reaches, a speed notebook leaves the
"Same features?" cell empty: there is no scikit-learn selection to
compare with.

The RFE, RFECV and Lasso results come from a plain script,
[`scripts/benchmark_speed_selectors.py`](../scripts/benchmark_speed_selectors.py),
against liblinear at its defaults, with nothing else running. Their
speed results come from this script, not from a notebook.

## D.2 Lasso on real SCAN-B

The script of Appendix D.1 also times Lasso on the five tasks of real
SCAN-B (2,976 patients, 28,552 genes;
[`results/speed/LassoSelector_scanb.csv`](../results/speed/LassoSelector_scanb.csv)):

| One Lasso fit per task | Seconds | Against liblinear |
|---|---|---|
| liblinear | 4.8 to 5.7 | — |
| PPDNA, GPU | 0.71 to 1.36 | 3.9 to 7.0 times faster |
| PPDNA, CPU | 4.5 to 31.3 | faster on Basal and Normal, up to 5.9 times slower on LumB |

The script standardizes the data in 32-bit floats, as the runner holds
it. The 11 flat genes of section 2.1 are nearly constant, with a spread
too small for 32-bit floats to hold, so standardizing them in 32 bits
turns them into noise. `load_xy` drops them (section 2.1).

## D.3 Why PPDNA, and how ours departs from the paper

Simpler solvers failed. FISTA, even on a working set of genes, only tied
liblinear. Parallel coordinate descent overshot on correlated genes.
SSNAL handles only the squared loss. Our PPDNA departs from the paper
(Liu and Zhou 2023) in five ways, each needed on SCAN-B-shaped data:

- The intercept is penalized like a weight, as liblinear penalizes it.
  This makes the problem the same as liblinear's.
- Its proximal step size starts at 40 C, the paper's start (σ₀ = 40/λ,
  section 4.1.2, with λ = 1/C), and grows five-fold per outer step (set
  by hand), up to 10⁶ C. Beyond that, rounding loses the digits the weights
  are made of.
- The Newton matrix is factored in 32-bit floats, where this GPU is fast.
  The solution is refined in 64 bits.
- The inner loop also stops when full Newton steps no longer shrink the
  gradient. Below that floor of 64-bit rounding, more steps only add
  noise.
- The outer loop also stops when three outer steps (a count set by
  hand) fail to halve the best residual.
  Rounding can set a floor above the target of 10⁻¹⁰; without this rule
  the solver would then run all 1,000 outer steps.

## D.4 Lasso: speed against liblinear

For Lasso the speed argument for dropping liblinear is weaker than for
RFE. On the CPU, a single PPDNA fit was faster than liblinear on Basal
and Normal, and up to 5.9 times slower on LumB (Appendix D.2). Its cost
grows with the number of genes the model keeps, and the luminal tasks
keep the most (section 6.2). A timing with C chosen by validation was
also made, but not saved, so it is not given here. Only on a GPU, which
liblinear cannot use, was PPDNA clearly faster (Appendix D.2).

Unlike every other wrapped selector, Lasso has no test that it is at most
5% slower than scikit-learn. No scikit-learn solver reaches Lasso's exact
solution. scikit-learn's Newton solvers refuse the L1 penalty. liblinear and saga
accept it but stop at a tolerance: less work, a different answer. On
synthetic data of 3,000 samples, PPDNA on the CPU was 1.6 to 7.1 times
slower than liblinear at its defaults, at 1,000 to 30,000 features
([`results/speed/LassoSelector.csv`](../results/speed/LassoSelector.csv)). So Lasso's speed comparison is the SCAN-B one
of Appendix D.2.

## D.5 The random forest on a GPU

A 4 GB GPU is small. Two GPU forests were tried, as single trials whose
times were not saved, so only their outcomes are given here:

- **XGBoost on the GPU** ran out of memory on SCAN-B-shaped data (1,488
  samples × 28,552 features), even with 64 bins. On a 3,000 × 30,000 set
  it also ran out of memory. Only on small data (3,000 × 3,000) did it
  run, and there it tied scikit-learn.
- **cuML** ran, but was slower than scikit-learn on the CPU. Its time went
  mostly to the middle levels of the trees, where nodes are many, not to
  memory.

Tree importance therefore stays on the CPU, with scikit-learn on several
threads (4 per fit in the experiments). A larger GPU is the next test for XGBoost (section 10.1).

## D.6 Libraries we did not use

JAX was rejected for the U test. It is a new dependency, reserves most
GPU memory per process, and recompiles for every data shape. CuPy and
Numba were not needed. Library sorts in PyTorch were enough, and PyTorch
is already needed for the embeddings.
