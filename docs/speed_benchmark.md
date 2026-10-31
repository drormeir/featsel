# Speed: the runner against plain scikit-learn

NEXT.md step 4, task 3. Design: `docs/experiment_design.md` 2e. Run
2026-10-31 with:

    python scripts/benchmark_speed.py --n-genes 3000 --out <file>

The same grid runs four ways: scikit-learn's `cross_validate` once per
selector and `k`, and the runner (`featsel.run`), each on one core and on
every core. The scikit-learn side uses scikit-learn's own selectors:
`SelectFromModel` with L1 logistic regression for Lasso, and `RFE` with
`LinearSVC`. The script checks that every test score is the same on all
four runs before it reports a time.

## Result

SCAN-B, Basal vs rest, 3000 random genes of 28552, 2976 patients. 5 splits at train fraction 0.5. Lasso and RFE (step 0.5) at k = 10, 50, 250, logistic regression with fixed C. 12 cores. All 30 test macro-F1 scores are equal on every run.

| run | seconds | speed-up over scikit-learn on 1 core |
|---|---|---|
| scikit-learn, 1 core | 33.6 | 1.0 |
| runner, 1 core | 10.4 | 3.2 |
| scikit-learn, 12 cores | 22.3 | 1.5 |
| runner, 12 cores | 6.3 | 5.3 |

Rerun 2026-10-31 with the data in float32 on both sides, as the runner now
holds it.

## What it shows

- **On one core the runner is 3.2 times faster.** scikit-learn refits
  imputation, scaling and the selector for every `k`. The runner prepares
  each split once and fits each selector once per split for all three `k`.
- **Parallel work adds little at this size.** scikit-learn spreads only the
  5 splits of one `cross_validate` call, 1.5 times faster. The runner has
  10 jobs (2 selectors times 5 splits), and starting the worker processes
  takes about 4.5 of its 6.3 seconds. 1.7 times faster than itself on one
  core.
- The scores are identical, so the speed comes from doing less work, not
  different work.

## Limits

- 3000 of 28552 genes, one task and 5 splits, to see the whole flow in about
  a minute. On full SCAN-B each fit is about ten times longer, so start-up
  weighs less and parallel work should pay more. Not measured here.
- Each selector and the classifier have a fixed seed, the same on every
  split, so both sides fit the same models. The experiments seed each split
  differently.
- C is fixed. The experiments tune it inside each split, which multiplies
  the classifier's cost on both sides alike.
