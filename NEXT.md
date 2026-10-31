# NEXT.md - the one thing being worked on

One entry. When it is done, replace it with the next one. Nothing else in this
file, and nothing else claims attention while it is open.

## Open

**The restart program: the smallest complete answer to each of the four
tasks, breadth first (`SCOPE.md` 5), then depth.**

Decided 2026-10-31. One step at a time; commit and push after each small
step; tick the step here when done.

Protocol decisions (recorded in `docs/experiment_design.md`): 5 stratified
Monte Carlo splits at train fraction 0.5, every filter scores genes by the
Mann-Whitney U test, C is tuned inside each split.

1. Fix the protocol on paper and in code.
   - [x] 1a. Rewrite `docs/experiment_design.md` to the decided protocol.
     Committed in `511f5aa`.
   - [x] 1b. Add `MannWhitneySelector` (top-k by U test) and use it in place
     of ANOVA in every experiment config. Committed in `8a6abfc`.
   - [x] 1c. Build nested tuning of C (`docs/experiment_design.md` 2b):
     inside each outer training half, inner stratified splits choose C for
     logistic regression and linear SVM from a small fixed grid; refit on the
     whole training half; score once on the test half. Then mark 2b as built.
     A model's `grid` in the config; 3 inner splits at 0.5, by ROC-AUC.
2. [x] Smallest complete selector comparison on SCAN-B (tasks 1-2): all
   selector entries of `configs/smoke_scanb.yaml`, k = 10, 50, 250,
   logistic regression with tuned C, the five subtype tasks, 5 splits. Write
   a config for it, run it, commit the config and a summary.
   `configs/comparison_scanb.yaml`, summary in `docs/comparison_scanb.md`.
   Along the way: RFE ranks once per task (about 11 times faster in the
   run), Lasso never selects a zero weight.
3. [ ] The same on BreastMNIST (task 4).
4. [ ] Measure the speedup (task 3), as in `docs/experiment_design.md` 2e:
   the runner against plain scikit-learn on a small grid.
5. [ ] Write each report chapter right after its step: Methodology after
   1, Results after 2-3, Implementation after 4. Methodology drafted in
   `docs/report/04_methodology.md`.
6. [ ] Only then, depth: more splits, more k, PAM50 gene recovery, HC
   behaviour, rerun `notebooks/02_classifier_behaviour.ipynb` with
   MannWhitneySelector (its config now names it; the committed figures
   still show the F-test ANOVA run).

Open question: a `demos/tutorial.ipynb` for MannWhitneySelector, which every
other selector has.

Not part of this program: the full SCAN-B grid (about four days). Park
anything else in `docs/ideas.md`.

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
