# NEXT.md - the one thing being worked on

One entry. When it is done, replace it with the next one. Nothing else in this
file, and nothing else claims attention while it is open.

## Due 2026-08-07

**One end-to-end run on SCAN-B.**

Load `configs/scanb_small.yaml`, run one selector (ANOVA F, top 100 genes),
train one classifier (logistic regression) under stratified 5-fold
cross-validation with PAM50 as a five-class target, print accuracy and
macro-F1.

Done means: the number is printed, the script is committed, and the output is
pasted into the weekly check-in. A bad score still counts. A crash that is
diagnosed and written down still counts.

Not part of this step: other selectors, other datasets, parallelization,
report chapters, refactoring. Park anything else in `docs/ideas.md`.

First move: `pytest tests/ -v`, to find out what still works.
