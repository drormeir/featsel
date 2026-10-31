# NEXT.md - the one thing being worked on

One entry. When it is done, replace it with the next one. Nothing else in this
file, and nothing else claims attention while it is open.

## Open

**Write the Methodology chapter, `docs/report/04_methodology.md`.**

The methods are now settled and running: binary subtype-vs-rest tasks,
stratified Monte Carlo splits shared by all tasks, selection inside the
split, the random control, the selectors including both Higher Criticism
variants and the U test, and the metrics. The chapter is missing; the
outline (`docs/report/00_report_outline.md`) lists it.

Done means: the chapter describes the selection methods, the evaluation
protocol and the metrics as the code runs them today, cites the sources in
`references/bibliography.md`, and is committed. Decisions already recorded
in `docs/experiment_design.md` and `docs/report/03_data.md` are referenced,
not repeated.

Not part of this step: new runs, the full SCAN-B grid, other chapters. Park
anything else in `docs/ideas.md`.

First move: list what `docs/experiment_design.md` says that no longer
matches the code, such as nested tuning, before writing.

## Done

- 2026-10-31 - The classifier-behaviour figures on binary tasks.
  `configs/experiment_classifiers.yaml` ran on the current runner: Random and
  ANOVA, k = 10 to 1000, five classifiers, five subtype tasks, 12 splits,
  4,200 executions in 26.5 minutes. The notebook shows the mean over tasks
  and a new per-subtype gain figure. Committed in `3d73968`.

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
