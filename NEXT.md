# NEXT.md - the one thing being worked on

One entry. When it is done, replace it with the next one. Nothing else in this
file, and nothing else claims attention while it is open.

## Open

**Regenerate the classifier-behaviour figures on binary tasks.**

The figures in `notebooks/02_classifier_behaviour.ipynb` show five-class
PAM50 results. Since 2026-10-18 every task is binary, each subtype vs the
rest (`SCOPE.md` 3a), so the figures are out of date.

Done means: the notebook reads a results CSV from the current runner, its
figures show the five subtype-vs-rest tasks, and the notebook is committed
with its outputs. A small run is enough; the full grid is not part of this
step.

Not part of this step: the full SCAN-B grid, the second dataset, report
chapters. Park anything else in `docs/ideas.md`.

First move: read the notebook and list which CSV and which columns it uses.

## Done

- 2026-10-31 - The wrapper and Higher Criticism selectors. `RFESelector`
  wraps scikit-learn's `RFE`. `HigherCriticismSelector` has a 2004 variant
  (Donoho and Jin 2004) and a 2008 variant with the exact null mean
  i/(N+1), scores genes by the Mann-Whitney U test, keeps p-values as logs,
  and has a tutorial. Both are in `configs/experiment_scanb.yaml`. The
  smoke test (`configs/smoke_scanb.yaml`) runs every selector and classifier
  end to end in about 3 minutes. The full grid was skipped by decision
  (about 4 days). alpha0 stays 0.1; HC2008 reaches its edge on SCAN-B, see
  `docs/report/03_data.md`. Along the way: binary tasks only, shared-memory
  splits, a per-worker memory cap and a clean exit when a worker dies.
- 2026-08-02 - One end-to-end run on SCAN-B (ANOVA F top-100, logistic
  regression, stratified 5-fold, PAM50 five-class): accuracy 0.8850, macro-F1
  0.8473. Committed in `149180f`.
- 2026-08-02 - Random-selection control, evaluation harness over the
  (selector x k x classifier x fold) grid, embedded selectors, and the
  classifier-behaviour notebook. Committed in `cc4f45a`, `8322e5f`, `4bac341`.
