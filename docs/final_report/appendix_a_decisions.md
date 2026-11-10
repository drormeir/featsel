# Appendix A. Decision log

[Back to contents](README.md)

The decisions that shaped the project, in date order. Each gives the
problem, the decision, and the alternatives rejected. Sources: `SCOPE.md`,
the working notes once in `docs/` and the drafts once in `docs/report/`
(both merged into this report and removed; git history keeps them), and
the project's working sessions. New decisions are added here.

## Software design

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-09-26 | How to structure the experiment code. | One runner; the protocol is a config (a YAML file or a Python dict); data always arrives as `X, y`. | Functions with many arguments (the YAML and Python forms would differ); pure scikit-learn `GridSearchCV` (no separate selection timing, no resume, and task 3 would shrink to setting `n_jobs`); dataset objects (they would only wrap `X, y`). |
| 2026-09-26 | The old selector wrapper silently dropped misspelled parameters. | Every selector is a scikit-learn estimator (`BaseEstimator`, `SelectorMixin`) and registers itself by class name. | A wrapper with a parameter whitelist. |
| 2026-09-26 | How to name selectors in configs and code. | Each class registers under its class name, case-insensitive, with optional aliases; Python users import the class; `create(name)` only for names read as text. | A hand-written name table (a second edit per selector); a short declared name per class (aliases already cover renames); `create` as the Python entry point (a typo fails late). |
| 2026-09-26 | Where selector code lives. | One folder per selector, with its tests and notebooks. | One file or folder per family (no shared code; unmanageable once a method has CPU and GPU versions). |
| 2026-09-26 | Whether to rewrite scikit-learn's methods. | Wrap them by default; write our own only when a timing shows a real gain, checked against scikit-learn's selection. | Reimplementing every method (most cost seconds while model fitting costs minutes). |
| 2026-09-29 | How to prove a selector is a valid drop-in replacement. | Two tests per wrapped selector: the same features as scikit-learn, and at most 5% slower. | Checking accuracy only. |
| 2026-09-30 | RFE dominates the cost. | Wrap scikit-learn's `RFE` first, to run the whole pipeline; save time by ranking once per split; an exact solver later (2026-11-07). | Writing RFE by hand first (a hand-written loop does the same work). |
| 2026-10-04 | Which classifier to judge selections with. | A simple one: L2-penalized logistic regression with a bias term. A simple classifier stresses the selection, not the model. | Tuning a complex classifier per method. |
| 2026-10-16 | How to load datasets that arrive in any shape. | One folder per dataset: a class that downloads and caches the published files unchanged, and a small `featsel_view.py` with this project's choices. | A generic CSV loader described by YAML (cannot foresee multi-table or image data). |
| 2026-10-16 | The deliverable. | Drop-in selectors for existing scikit-learn pipelines; the runner is the research harness. | A standalone pipeline that users would have to adopt whole. |
| 2026-10-16 | What a dataset loader returns. | A pandas DataFrame and Series, so names reach the selectors and labels cannot be misaligned. | NumPy arrays and a name list (GSE96058's two files list samples in different orders). |
| 2026-10-17 | Where a dataset's choices live. | In `featsel_view.py`; the experiment config names the module. | A dataset YAML naming the loader (it repeated the choices, so the two could drift); moving `k` into the view (the config would no longer show the whole experiment). |
| 2026-10-24 | Long runs died silently when memory ran out. | A memory cap per worker turns exhaustion into a Python `MemoryError`; the cell is skipped and the run goes on. Later: resume, and restart with fewer workers. | Running with fewer workers always (wastes the machine). |
| 2026-10-31 | The runner refit rank-based selectors for every `k`. | A class flag `rank_once`: fit once per split, cut the ranking at each `k`. | A list of class names kept in the runner. |
| 2026-11-01, 2026-11-07 | Reproducibility with several random generators. | `random_state` defaults to 42; every selector seeds Python's, NumPy's and PyTorch's generators at the start of `fit`. | A generator stored in the instance (scikit-learn's `clone` breaks it); passing one shared generator (breaks repeatability). |
| 2026-11-09 | Per-feature scores all lived under `feature_importances_`, with different meanings. | Each selector names its measurement (`scores_`, `log_pvalues_`, `coef_`, `variances_`); `ranking()` only where a cut equals a fit. | One shared attribute name. |

## Data

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-10-16 | The course's copies of SCAN-B had an unknown gene selection and repeat samples. | Load GSE96058 directly; keep HiSeq 2000 samples only; one sample per patient. Retire the course copies (archived outside git). | Using the course copies; mixing the two sequencing machines. |
| 2026-10-16 | PAM50 and ER are computed from gene expression. | Say so in the report, and test whether selection finds the 50 PAM50 genes. | Treating them as clinical facts, as an earlier course report did. |
| 2026-10-17 | Task 4 needs a second dataset. | BreastMNIST, as ResNet-50 embeddings: public, not gene expression, more features than samples. | Text (TF-IDF), UCI tabular sets, the NIPS 2003 challenge sets; raw pixels (a CNN beats feature selection there). |
| 2026-10-18 | PAM50 has five imbalanced classes. | Binary tasks only: each subtype against the rest. | A five-class task; the union of per-class gene sets. Recombining the binary classifiers was parked. |

## Statistics and protocol

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-08-02 | Whether to vary preprocessing. | Standardization only, fit inside the split; a quantile transform as a side experiment (cancelled). | Min-max scaling (one outlier squeezes the rest); preprocessing as a full axis (doubles the grid). |
| 2026-09-26 | k-fold scores are correlated and too few. | Stratified Monte Carlo splits; median with quartiles. | k-fold; leave-one-out (no AUC per split, one refit per patient). |
| 2026-09-26 | Tuning C on the test part would leak. | Nested tuning inside each training part. Built 2026-10-31. | Fixed C (kept only for the frontier, to save time). |
| 2026-09-26 | A new metric means a new run. | Store each test patient's label and score, and derive every metric afterwards. Designed, not built. | Storing only the metrics (what the runner does today). |
| 2026-09-26 | How to set the feature budget. | `k` as a percent of the samples, plus `k` = 50 for PAM50; below the training size. Later given directly. | A single `k` (cannot tell small-budget methods from large-budget ones). |
| 2026-10-12 | Which score judges a selection. | AUC: no threshold, no cost assumption. Macro-F1 for the comparison's tables, for imbalance. | F1 alone (assumes a cost trade-off); accuracy (misleading on imbalanced tasks). |
| 2026-10-13 | One number for a method's strength. | Report curves over `k`, scored per split by AUC. A leave-one-out confusion table was built (`featsel/strength.py`) but not used. | Area under the score-against-`k` curve (hides where a method wins); leave-one-out (no spread, needs a threshold). |
| 2026-10-24 | p-values below 10⁻³⁰⁸ round to zero. | Compute log p-values directly. | 32-bit floats (underflow even sooner); wider floating-point types (the log removes the need). |
| 2026-10-25, 2026-10-31 | The F-test assumes normal values; ours are tied at a floor on an arbitrary scale. | Every filter in the experiments uses the Mann-Whitney U test. | The F-test (kept only in the scikit-learn drop-in selectors); a four-parameter floor model (out of scope). |
| 2026-10-05, 2026-10-24 | Which Higher Criticism. | Two forms: Donoho and Jin 2004 as published, and their 2008 thresholding with the exact null mean i/(N+1) in place of the papers' i/N. | Unpublished variants (exact variance, Berk-Jones as a selection rule): parked. |
| 2026-10-31 | HC2008 stops at its search edge on SCAN-B. | Keep alpha0 = 0.1, as in both papers. | Raising alpha0 because we know the signal is widespread (that tunes the method to the answer). |
| 2026-10-31 | The full grid takes about four days. | First 5 splits, then 16 on SCAN-B and 20 on BreastMNIST, at one train fraction (0.5); float32 data; 50 trees; RFE half the genes per round. | 100 splits at three train fractions. |
| 2026-10-31 | Which test the filters use. | Mann-Whitney replaces ANOVA in every experiment config. | Keeping the F-test (cheaper, about 1 s against 10 s, but its assumptions fail here). |
| 2026-11-07 | Pearson correlation with a gene on an arbitrary scale. | Spearman (rank) correlation in the frontier. | Pearson (its value changes with the transform). |
| 2026-11-07 | What to time, how often. | Selection time only; median of 10 fits per selection; the device recorded. | One fit (noisy); including classifier time. |

## Speed and hardware

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-11-03 | The U test took about 10 s per fit. | A NumPy version with one sort per gene, then a GPU version in PyTorch, `device='auto'` by default; identical results on every device. | JAX, CuPy and Numba (one GPU library is enough; PyTorch is already the embedding dependency). |
| 2026-11-07 | How far to optimize. | Every selector as fast as it can be on this machine; scikit-learn's version stays the reference and the default wherever a faster one can select differently. | Optimizing only the slowest. |
| 2026-11-07 | Single timings mislead. | Speed claims only from a benchmark series on synthetic data shaped like SCAN-B; AUC only on real data. | Quoting one timing. |
| 2026-11-07 | What the study should deliver to practitioners. | A cost-quality frontier: selection time against AUC, every method on one plot. | Accuracy tables only. |
| 2026-11-08 | Mutual information on the GPU must stay exact. | The whole estimate on the GPU; only scikit-learn's random noise drawn on the CPU and sent over. | Drawing noise on the GPU (different numbers from scikit-learn). |
| 2026-11-08 | Random forest on the GPU. | Not done: XGBoost ran out of memory at full size on 4 GB, and cuML was 3 times slower. Forest stays on the CPU, all threads. | Shipping a slower GPU path. |
| 2026-11-08 | RFE on this GPU. | RFE's exact solver stays on the CPU: it needs 64-bit floats, which this GPU runs slowly. | 32-bit floats on the GPU (changed the genes selected). |
| 2026-11-09 | The frontier's design. | Train on 70% of the patients; GPU methods with one worker and `device='cuda'`; CPU methods with two workers; Sequential with one fit per selection. | Training on 50% (less realistic); `device='auto'` (would fall back to the CPU while the GPU is busy). |

## Report

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-11-09 | The earlier draft in `docs/report/` was dense and written for the author. | This report, for B.Sc.-level readers, in `docs/final_report/`; figures copied in so links work on GitHub. | Rewriting `docs/report/` in place (it keeps generated tables). |
| 2026-11-09 | Are differences between methods real? | Paired Wilcoxon signed-rank tests over splits, with a Bonferroni threshold. | Comparing medians by eye. |
| 2026-11-10 | Content in `docs/report/` and `docs/*.md` was missing from this report. | Merge it in: related work into chapter 1, data plans into Appendix D, full comparison tables into Appendix E, design and protocol details into chapters 3 to 7. Then remove `docs/report/`; its numbers came from finished runs, so the generated tables were not carried over. Then remove the working notes in `docs/` too: this report becomes the only design record, Appendix A the decision log, section 10.3 the parked ideas. | Linking to the working notes only (a reader would need both); keeping `docs/report/` for its generated tables; keeping the notes as a second record (two records drift apart). |
