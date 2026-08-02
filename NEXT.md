# NEXT.md - the one thing being worked on

One entry. When it is done, replace it with the next one. Nothing else in this
file, and nothing else claims attention while it is open.

## Due 2026-08-12

**The wrapper and Higher Criticism selectors.**

Add `RFESelector` in `selectors/wrapper.py` and `HigherCriticismSelector` in
`selectors/higher_criticism.py`, both behind the existing `BaseSelector`
interface and registered in `FeatureSelector._METHOD_MAP`. Higher Criticism
follows the published Donoho and Jin thresholding formulation: per-feature
p-values from a univariate test, sorted, HC statistic, threshold. It selects
its own feature count, so `n_features` is ignored for it.

Done means: both appear in `SELECTORS` in `featsel/experiment.py`, the grid
runs end to end with all six selectors, and `results/` plus the figures in
`notebooks/02_classifier_behaviour.ipynb` are regenerated and committed. A bad
score still counts.

Not part of this step: the second dataset, parallelization, per-class task
framing, G-mean, report chapters. Park anything else in `docs/ideas.md`.

First move: read the Donoho and Jin formulation in `references/` before
writing any code.

## Done

- 2026-08-02 - One end-to-end run on SCAN-B (ANOVA F top-100, logistic
  regression, stratified 5-fold, PAM50 five-class): accuracy 0.8850, macro-F1
  0.8473. Committed in `149180f`.
- 2026-08-02 - Random-selection control, evaluation harness over the
  (selector x k x classifier x fold) grid, embedded selectors, and the
  classifier-behaviour notebook. Committed in `cc4f45a`, `8322e5f`, `4bac341`.
