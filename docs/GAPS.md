# Gaps to close for the final grade

Written 2026-11-10. The grade rests on two things, from the project
proposal (`references/MLDS_Feature_Selection.pdf`): the four tasks, and the
final report's academic quality. Every gap here serves one of them.

Each gap names its problem, why it matters for the grade, and what closes
it. "Source" says where the gap is already recorded. "New" means the final
report does not list it yet.

## A. Gaps in the evidence for the tasks

### A1. The parallel speedup is not measured at full size (task 3)

- **Problem.** Parallel work was measured only on a small grid: 3,000
  genes, 5 splits. There it gave 1.7 times, and starting the workers took
  4.5 of 6.3 seconds. On full SCAN-B it was never measured.
- **Why it matters.** Task 3 asks for parallelized feature selection. The
  evidence for the parallel part is the weakest of the four tasks.
- **What closes it.** A scaling curve on full SCAN-B: the same grid with 1,
  2, 4, 6 and 12 workers. Report time, speedup and efficiency per worker
  count. Repeat each point a few times.
- **Source.** `final_report/06_making_it_fast.md`, 6.1, lines 54-59. Not in
  chapter 10. New as a gap.

### A2. The speed notebooks' overlap column is wrong at large sizes (task 3)

- **Problem.** Past the original's largest size, the overlap is measured
  against the first variant that ran, not the original. This affects
  RFECV, tree importance and mutual information at 30,000 genes, and
  Sequential at 3,000 and 10,000 genes.
- **Why it matters.** "Same features as scikit-learn" is part of the task 3
  evidence. Some of those numbers are wrong.
- **Source.** `final_report/10_open_work.md`, 10.1.

### A3. Does selection beat using all features? Not tested (task 2)

- **Problem.** On SCAN-B the best selected genes beat all genes by 0.006
  AUC, over 5 splits, untested. On BreastMNIST and FGVC-Aircraft all
  features score best.
- **Why it matters.** This is the central question of the comparison. The
  current answer is "barely, maybe".
- **What closes it.** Add the all-features baseline to the 16-split
  comparison, and test it with the corrected test of B1.
- **Source.** `final_report/10_open_work.md`, 10.1.

### A4. The frontier rests on 5 splits (task 2)

- **Problem.** On 5 splits the corrected paired test (B1) has 4 degrees
  of freedom. So the order of the top methods on the frontier is not
  established.
- **What closes it.** More splits: about 8 hours per 5 on SCAN-B.
- **Source.** `final_report/10_open_work.md`, 10.1.

### A5. BreastMNIST holds copied images (task 4)

- **Problem.** 155 of 780 source images are copies, some labelled
  differently. Copies on both sides of a split leak, by an unmeasured
  amount.
- **Why it matters.** Every task 4 result carries the leak.
- **What closes it.** A loader option that drops the copies, and two
  reruns. Minutes.
- **Source.** `final_report/10_open_work.md`, 10.1.
- **Status.** Closed 2026-11-11. The letter's list names BUSI files, which
  BreastMNIST does not keep. So `copies.py` finds the copies in
  BreastMNIST's own images by the letter's method, geometric registration:
  150 groups of 339 images, against the letter's 153 and 346, and the same
  6 groups with conflicting labels. `load_xy(drop_copies=True)` keeps 585
  images. The comparison and the frontier ran again without copies.
  Scores did not fall by a visible amount; the fine ranking changed
  (sections 2.2 and 7.8).

### A6. Berk-Jones is not among the methods (task 2)

- **Problem.** Berk-Jones is a close relative of Higher Criticism. It
  compares each sorted p-value with its exact null distribution: under
  the null, the i-th smallest of N p-values follows Beta(i, N − i + 1).
  Its score is that Beta CDF at the observed p-value. The threshold is
  where the score is smallest, so it picks its own count, like HC.
- **Why it matters.** It adds a method to the comparison, in the same
  family as HC, from the same p-values.
- **Existing code.** The thesis repository,
  `/home/dror/projects/RUNI/Thesis`:
  - `python/Higher_Criticism.py`, class `Berk_Jones`: the score
    `beta.cdf(p_sorted, i, N - i + 1) - 0.5`, minimized over a search
    fraction `gamma`.
  - `python/adaptive_methods/python_native.py`, `berk_jones`: the same
    Beta CDF with SciPy.
  - `python/adaptive_methods/numba_cpu.py` and `numba_gpu.py`: Numba
    versions on the CPU and the GPU, using a cached log-gamma table.
  - `tests/berk_jones_test.ipynb` and
    `python/adaptive_methods/bitexact_adaptive_methods.ipynb`: checks.
- **What closes it.** A `BerkJonesSelector` beside
  `HigherCriticismSelector`. It would take the same U-test log p-values
  and the same search limit. Our p-values can fall below 1e-308, so the
  Beta CDF must be computed in log form.
- **Source.** Parked on 2026-10-05 as "published only as a detection test,
  not for feature selection". Reversed on 2026-11-10: `DECISIONS.md`
  records the decision and no longer lists it as parked.
- **Status.** Closed 2026-11-11. `BerkJonesSelector`, with the CDF in log
  form, tested against SciPy and 50-digit arithmetic. Run on the
  comparison's splits and on the frontier. It stopped at or just before
  HC's search edge, kept a subset of HC2008's features on every split, and
  scored the same (sections 4.5 and 7.5).

### A7. No two-stage pipeline is tested (task 2)

- **Problem.** Every experiment runs one selector alone. A common practice
  chains them: a fast filter cuts the features first, and a slower method
  picks the final set. The report never tests such a chain.
- **Why it matters.** Reviewers treat such hybrids as standard (Saeys et
  al. 2007).
- **What closes it.** Add chains to the frontier, such as Mann-Whitney to
  1,000 features, then RFE or Lasso. Compare each chain's time and AUC
  with the same method alone. The runner's configs cannot express a chain
  today: each entry is one selector. Either the runner gains a list of
  steps, or a selector class wraps a scikit-learn `Pipeline` of two
  selectors.
- **Source.** New.

## B. Gaps in the report's academic quality

### B1. The significance test overstates certainty

- **Problem.** Chapter 7 compares methods with the Wilcoxon signed-rank
  test over 16 random splits. The splits share most of their training
  patients, so the per-split scores are not independent. A test that
  treats them as independent gives p-values that are too small (Nadeau and
  Bengio 2003).
- **Why it matters.** The "real" differences in chapter 7 rest on these
  p-values.
- **What closes it.** The corrected resampled t-test of Nadeau and Bengio.
  Report effect sizes with intervals, not only p-values. Re-run
  `scripts/paired_tests.py` with the corrected test; no new experiment is
  needed.
- **Source.** New.
- **Status.** Closed 2026-11-11. `scripts/paired_tests.py` runs the
  corrected test, with the mean difference and its 95% interval. Section
  7.3 is rewritten. Tests passed on SCAN-B fell from 67 to 35 of 81, and on
  BreastMNIST from 30 to 1 of 61. Tree importance's lead over Mann-Whitney
  is no longer established; chapters 7 and 9 and the summary say so.

### B2. The stability index assumes equal set sizes

- **Problem.** Kuncheva's index is defined for sets of equal size. Lasso,
  correlation, HC and RFECV choose different sizes on different splits,
  so their stability is approximate. Nogueira, Sechidis and Brown (2018)
  give a measure that handles varying sizes.
- **What closes it.** Add their measure next to Kuncheva's. Computed from
  existing results; no new runs.
- **Source.** New. The aircraft notebook notes the approximation.
- **Status.** Closed 2026-11-11. `featsel.analysis.nogueira_stability`,
  tested to equal the mean Kuncheva index at equal sizes.
  `scripts/summarize_comparison.py` prints both. On the comparison it
  changes no value by more than 0.014 (HC2004). It matters for RFECV on
  the aircraft data: 0.15 against Kuncheva's 0.07. Sections 3.6 and 7.4,
  Appendix D and the aircraft notebook say so.

### B3. No stated contribution

- **Problem.** The report has no list of what is new compared with earlier
  work. Section 1.6 lists what earlier work found, but not what this
  project adds.
- **What closes it.** A short list in chapter 1, mirrored in chapter 9.
- **Source.** New.
- **Status.** Closed 2026-11-11. Section 1.7 lists five contributions;
  section 9.3 repeats them in short.
