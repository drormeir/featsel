# Appendix A. Decision log

[Back to contents](README.md)

The decisions that shaped the project, in date order. Each gives the
problem, the decision, and the alternatives rejected. A row with two dates
is placed by its first. The first date is when the problem was raised or
decided; the second is when it was built or settled. Sources: `SCOPE.md`,
git history, the project's working sessions, the working notes once in
`docs/` and the drafts once in `docs/report/`. The notes and drafts were
merged into this report and removed. Git history keeps them. New decisions
are added here.

## Software design

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-09-26 | How to structure the experiment code. | One runner. The protocol is a config: a YAML file or a Python dict. Data always arrives as `X, y`. | Functions with many arguments (the YAML and Python forms would differ); pure scikit-learn `GridSearchCV` (no separate selection timing, no resume, and task 3 would shrink to setting `n_jobs`); dataset objects (they would only wrap `X, y`). |
| 2026-09-26 | The old selector wrapper silently dropped misspelled parameters. | Every selector is a scikit-learn estimator (`BaseEstimator`, `SelectorMixin`) and registers itself by class name. | A wrapper with a parameter whitelist. |
| 2026-09-26, 2026-09-28 | How to name selectors in configs and code. | Each class registers under its class name, case-insensitive, with optional aliases. Python users import the class. `create(name)` only for names read as text. | A hand-written name table (a second edit per selector); a short declared name per class (aliases already cover renames); `create` as the Python entry point (a typo fails late). |
| 2026-09-26, 2026-09-28 | Whether to rewrite scikit-learn's methods. | Wrap them by default. Write our own only when a timing shows a real gain, checked against scikit-learn's selection. | Reimplementing every method (most cost seconds while model fitting costs minutes). |
| 2026-09-28 | Where selector code lives. | One folder per selector, with its tests and notebooks. | One file or folder per family (no shared code; unmanageable once a method has CPU and GPU versions). |
| 2026-09-29, 2026-10-04 | How to prove a selector is a valid drop-in replacement. | Two tests per wrapped selector: the same features as scikit-learn, and `fit` plus `get_support` at most 5% slower. Each side's time is the median of 50 fits. The test data is large enough (1,000 × 4,000) that timing noise stays below 5%. | Timing `fit` alone (scikit-learn builds its mask later, in `get_support`); 10 fits (two equal versions drifted 5% apart); checking accuracy only. |
| 2026-09-30, 2026-10-05 | RFE dominates the cost. | Wrap scikit-learn's `RFE` first, to run the whole pipeline. Save time by ranking once per split. An exact solver later (2026-11-07). | Writing RFE by hand first (a hand-written loop does the same work). |
| 2026-10-04, 2026-10-12 | Which classifier to judge selections with. | A simple one: L2-penalized logistic regression with a bias term. A simple classifier stresses the selection, not the model. | Tuning a complex classifier per method. |
| 2026-10-16 | How to load datasets that arrive in any shape. | One folder per dataset: a class that downloads and caches the published files unchanged, and a small `featsel_view.py` with this project's choices. | A generic CSV loader described by YAML (cannot foresee multi-table or image data). |
| 2026-10-16 | The deliverable. | Drop-in selectors for existing scikit-learn pipelines. The runner is the research harness. | A standalone pipeline that users would have to adopt whole. |
| 2026-10-16 | What a dataset loader returns. | A pandas DataFrame and Series, so names reach the selectors and labels cannot be misaligned. | NumPy arrays and a name list (GSE96058's two files list samples in different orders). |
| 2026-10-17 | Where a dataset's choices live. | In `featsel_view.py`. The experiment config names the module. | A dataset YAML naming the loader (it repeated the choices, so the two could drift); moving `k` into the view (the config would no longer show the whole experiment). |
| 2026-10-17, 2026-10-31 | The runner refit rank-based selectors for every `k`. | Fit once per split and cut the ranking at each `k`: Lasso and tree importance first. Selection per split fell from about 245 s to 39 s, with identical genes. Since 2026-10-31 a class flag, `rank_once`, marks such selectors, and RFE is one of them. | A list of class names kept in the runner (the first version). |
| 2026-10-24 | Long runs died silently when memory ran out. | A memory cap per worker turns exhaustion into a Python `MemoryError`. The cell is skipped and the run goes on. Resume already existed. Since 2026-11-07 the cap counts data memory, and a run restarts by itself with fewer workers when one dies. | Running with fewer workers always (wastes the machine); a system-wide memory limit (every program on the machine is affected); a service memory limit (the kernel still kills, with no `MemoryError`); lowering a worker's priority for the kernel's out-of-memory killer (it protects the editor, but the cell still dies). |
| 2026-10-24 | Each worker held its own prepared copy of the data, 2 to 3 GB. | One split, stratified on the five-class PAM50 label, serves all five tasks. Each split is prepared once into shared memory (`/dev/shm`). A job is one task and one selector. | Preparing the data per task and per worker (five times the preprocessing, and the memory of every copy). |
| 2026-10-31 | A review found the main comparison missing, the filters using different tests, and the design promising more than the code did. | Restart breadth first: fix the protocol, then the smallest comparison on SCAN-B, then BreastMNIST, then the speed-up, then the chapters, and only then depth. | Deepening one part before every task had a first answer. |
| 2026-10-31 | Lasso filled its top `k` with genes of weight zero. | Keep only genes with a non-zero weight, even when that is fewer than `k`. | Padding with zero-weight genes (their order is arbitrary). |
| 2026-10-31 | A fast RFE built on `SVC(kernel='linear')` passed every test, but chose useless genes on SCAN-B: macro-F1 0.48 on Basal at k = 10, against 0.95. Its weights came from only 177 of 1,488 training patients, the support vectors. | Return to `LinearSVC`, ranking once for every `k`. | The kernel SVM. The lesson: matching a reference implementation does not show that a method suits the data. |
| 2026-10-31 | Report numbers were typed by hand. | Results are tracked in git, and the report's facts are computed by scripts (`scripts/data_facts.py`). Recomputing corrected several values, such as 96.7% to 98.1% non-normal genes. | Typing numbers from notebooks. |
| 2026-11-01, 2026-11-09 | Reproducibility with several random generators. | `random_state` defaults to 42 and must be an integer. Every selector that uses randomness seeds Python's, NumPy's and, once imported, PyTorch's generators at the start of `fit`. Since 2026-11-09 this goes through a middle class, `FeatureSelectorRandom`. Selectors without randomness take no seed. Unseeded runs (`random_state=None`) were removed. | A generator stored in the instance (scikit-learn's `clone` breaks it); passing one shared generator (breaks repeatability); a class attribute for the seed (it must belong to the object); a seed parameter on every selector (ANOVA has no randomness). |
| 2026-11-09 | Every selector repeated the seeding and checks in its own `fit`. | The base `fit` seeds, then calls an abstract `_fit` that each selector implements. A selector without `_fit` fails when it is created. | Wrapping each class's `fit` when it is defined (rewrites classes at run time); raising `NotImplementedError` (fails only when `fit` runs). |
| 2026-11-09 | Per-feature scores all lived under `feature_importances_`, with different meanings. | Each selector names its measurement (`scores_`, `log_pvalues_`, `coef_`, `variances_`). `ranking()` only where a cut equals a fit. | One shared attribute name. |
| 2026-11-10 | Base-class helpers read attributes such as `self.n_features` that the base class never declared. | Helpers take what they read as arguments. A reader should understand a selector from its own file. | Implicit attributes shared through `self`. |

## Data

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-10-05, 2026-10-18 | PAM50 has five imbalanced classes. | Binary tasks only: each subtype against the rest. First run this way on 2026-10-05; made the project's rule on 2026-10-18. | A five-class task; the union of per-class gene sets. Recombining the binary classifiers was parked. |
| 2026-10-16, 2026-10-17 | The course's copies of SCAN-B had an unknown gene selection and repeat samples. | Load GSE96058 directly. Keep HiSeq 2000 samples only, one sample per patient. On 2026-10-17, retire the course copies (archived outside git). | Using the course copies; mixing the two sequencing machines. |
| 2026-10-16, 2026-10-17 | PAM50 and ER are computed from gene expression. | Say so in the report. On 2026-10-17, decide to test whether selection finds the 50 PAM50 genes. | Treating them as clinical facts, as an earlier course report did. |
| 2026-10-17 | Task 4 needs a second dataset. | BreastMNIST, as ResNet-50 embeddings: public, not gene expression, more features than samples. | Text (TF-IDF); UCI tabular sets; the NIPS 2003 challenge sets (Gisette has fewer features than samples, and its probe features are withheld; Madelon has only 500 features; Arcene has only 200 samples); raw pixels (a CNN beats feature selection there). |

## Statistics and protocol

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-08-02, 2026-10-05 | Whether to vary preprocessing. | Standardization only, fit inside the split. A rank-to-normal transform was compared on 2026-10-05 and did not help (section 3.10). | Min-max scaling (one outlier squeezes the rest); preprocessing as a full axis (doubles the grid). |
| 2026-08-02 | k-fold scores are correlated and too few. | Stratified Monte Carlo splits. Median with quartiles. | k-fold; leave-one-out (no AUC per split, one refit per patient). |
| 2026-09-26 | Tuning C on the test part would leak. | Nested tuning inside each training part. Built 2026-10-31. | Fixed C (kept only for the frontier, to save time). |
| 2026-09-26 | A new metric means a new run. | Store each test patient's label and score, and derive every metric afterwards. Designed, not built. | Storing only the metrics (what the runner does today). |
| 2026-09-26 | How to set the feature budget. | `k` as a percent of the samples, plus `k` = 50 for PAM50. Below the training size. Later given directly. | A single `k` (cannot tell small-budget methods from large-budget ones). |
| 2026-10-05, 2026-10-24 | p-values below 10⁻³⁰⁸ round to zero. Found on 2026-10-05: on the course data, 167 genes had an F-test p-value of exactly 0. | Compute log p-values directly. Built on 2026-10-24. | 32-bit floats (underflow even sooner); wider floating-point types (the log removes the need). |
| 2026-10-05, 2026-10-18 | Which Higher Criticism. | Two forms. HC2004 exactly as published: its formula is already exact, since it standardizes the share of p-values below a threshold. HC2008 with the exact null mean i/(N+1) in place of the papers' i/N. Settled on 2026-10-18, built on 2026-10-24. | The same correction in HC2004 (not needed there); HC2004's own variant HC+, which skips the smallest p-values (not used); Berk-Jones as a selection rule (published only as a test): parked. |
| 2026-10-10, 2026-10-12 | Which score judges a selection. | AUC: no threshold, no cost assumption. The runner recorded AUC only from 2026-11-07, so the comparison of 2026-10-31 reports macro-F1, which weights both classes equally. | F1 alone (assumes a cost trade-off); accuracy (misleading on imbalanced tasks). |
| 2026-10-13 | One number for a method's strength. | Report curves over `k`, scored per split. A leave-one-out confusion table was built (`featsel/strength.py`) but not used. | Area under the score-against-`k` curve (hides where a method wins); leave-one-out (no spread, needs a threshold). |
| 2026-10-24, 2026-10-31 | The full grid takes about four days. | Skip it (2026-10-24). Then first 5 splits, then 16 on SCAN-B and 20 on BreastMNIST, at one train fraction (0.5). float32 data. 50 trees. RFE removes half the genes per round. | 100 splits at three train fractions. |
| 2026-10-25, 2026-10-31 | The F-test assumes normal values. Ours are tied at a floor on an arbitrary scale. | Every filter in the experiments that runs a statistical test uses the Mann-Whitney U test. Higher Criticism moved to it on 2026-10-25; every filter on 2026-10-31. | The F-test (kept only in the scikit-learn drop-in selectors); Welch's t-test (still assumes normal values); a permutation test (its smallest p-value, about 1 over the number of shuffles, is far too coarse for HC); a four-parameter floor model (out of scope). |
| 2026-10-31 | HC2008 stops at its search edge on SCAN-B. | Keep alpha0 = 0.1, as in the 2008 and 2009 papers. | Raising alpha0 because we know the signal is widespread (that tunes the method to the answer). |
| 2026-10-31 | A draft claimed HC2008 scored best, but it used about 2,855 genes against at most 250 for the others. | Also run random selection and Mann-Whitney at HC2008's own count (2,855 on SCAN-B, 204 on BreastMNIST). | Comparing methods at different sizes. |
| 2026-10-31 | Which test the filters use. | Mann-Whitney replaces ANOVA in every experiment config. | Keeping the F-test (cheaper, about 1 s against 10 s, but its assumptions fail here). |
| 2026-11-06, 2026-11-07 | Pearson correlation with a gene on an arbitrary scale. | Spearman (rank) correlation in the frontier. | Pearson (its value changes with the transform). |
| 2026-11-07 | What to time, how often. | Selection time only. Median of 10 fits per selection. The device recorded. | One fit (noisy); including classifier time. |
| 2026-11-10 | `SCOPE.md` allowed published Higher Criticism only, but HC2008 uses i/(N+1) where the papers use i/N. | Keep i/(N+1), say so in section 4.4, and drop the rule from `SCOPE.md`. | Switching HC2008 to i/N (its results in chapters 7 and 8 would need a rerun). |

## Speed and hardware

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-11-02, 2026-11-03 | The U test took about 10 s per fit. | A NumPy version with one sort per gene: about 3 s. Then a GPU version in PyTorch, `device='auto'` by default. Identical results on every device. `SCOPE.md` allowed PyTorch in the package for this one method, because the three U-test selectors took about 37% of the comparison's computing time. | SciPy's private one-sort ranking (a private interface can break on upgrade); JAX, CuPy and Numba (one GPU library is enough; PyTorch is already the embedding dependency). |
| 2026-11-06, 2026-11-07 | How far to optimize. | Every selector as fast as it can be on this machine. `SCOPE.md` widened to allow a GPU path for every selector. scikit-learn's version stays the reference and the default wherever a faster one can select differently. Only problems with more features than samples are optimized. | Optimizing only the slowest. |
| 2026-11-06, 2026-11-07 | Single timings mislead. | Speed claims only from a benchmark series on synthetic data shaped like SCAN-B. AUC only on real data. | Quoting one timing. |
| 2026-11-06, 2026-11-07 | What the study should deliver to practitioners. | A cost-quality frontier: selection time against AUC, every method on one plot. | Accuracy tables only. |
| 2026-11-07 | Mutual information on the GPU must stay exact. | The whole estimate on the GPU. Only scikit-learn's random noise is drawn on the CPU and sent over. Divisions and sums follow NumPy's rounding and order. | Drawing noise on the GPU (different numbers from scikit-learn); copying the data back to the CPU for scaling and noise (the work must stay on the GPU). |
| 2026-11-07 | Lasso on the GPU. | PPDNA (Liu and Zhou 2023), a semismooth Newton method, run per task. | FISTA on all genes (21 s, about 3,000 steps); FISTA on a working set (2.2 to 3.1 s, only a tie with liblinear); parallel coordinate descent (correlated genes overshoot); SSNAL (squared loss only); solving the five PAM50 tasks in one batch (each task is treated as its own dataset). |
| 2026-11-07 | RFE on this GPU. | RFE's exact solver runs in 64-bit floats. At full size it is faster on the CPU, since this GPU runs 64-bit floats slowly. | 32-bit floats on the GPU (changed the genes selected). |
| 2026-11-07, 2026-11-08 | Random forest on the GPU. | Not done. XGBoost ran out of memory at full size on 4 GB. cuML was 3 times slower. The forest stays on the CPU, with all threads (`n_jobs`). | Shipping a slower GPU path. |
| 2026-11-08 | The frontier's design. A first frontier on 2026-11-07, at 50% training and `device='auto'`, ran 121 of its 125 Mann-Whitney and HC2008 fits on the CPU, because the GPU was busy. | Train on 70% of the patients. GPU methods with one worker and `device='cuda'`. CPU methods with two workers. Sequential with one fit per selection (ten would take about 24 hours). Run on 2026-11-09. | Training on 50% (less realistic); `device='auto'` (falls back to the CPU while the GPU is busy). |

## Report

| Date | Problem | Decision | Rejected |
|---|---|---|---|
| 2026-11-09 | The earlier draft in `docs/report/` was dense and written for the author. | This report, for B.Sc.-level readers, in `docs/final_report/`. Figures copied in so links work on GitHub. | Rewriting `docs/report/` in place (it keeps generated tables). |
| 2026-11-09 | Are differences between methods real? | Paired Wilcoxon signed-rank tests over splits, with a Bonferroni threshold. | Comparing medians by eye. |
| 2026-11-10 | Content in `docs/report/` and `docs/*.md` was missing from this report. | Merge it in: related work into chapter 1, data plans into Appendix D, full comparison tables into Appendix E, design and protocol details into chapters 3 to 7. Then remove `docs/report/`. Its numbers came from finished runs, so the generated tables were not carried over. Then remove the working notes in `docs/` too. This report becomes the only design record, Appendix A the decision log, section 10.3 the parked ideas. | Linking to the working notes only (a reader would need both); keeping `docs/report/` for its generated tables; keeping the notes as a second record (two records drift apart). |
| 2026-11-10 | The reference list held works cited only in passing. | Keep only works the report relies on. Twelve were removed, with their citations. | Keeping every work ever consulted. |
