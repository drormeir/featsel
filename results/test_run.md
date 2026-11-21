# Test run, 2026-11-21

Command, from the repository root:

    .venv/bin/python -m pytest -q -p no:cacheprovider

Code: commit 74c5479, with a clean `featsel/`, `featsel_datasets/` and
`tests/`.

Machine: Intel i7-9750H, 16 GB, NVIDIA GTX 1650 (4 GB), so the GPU tests
ran. No other job ran during the tests.

Result, the last line of pytest's output:

    413 passed, 5 warnings in 1360.27s (0:22:40)

No test failed or was skipped. The 10 speed tests
(`test_not_slower_than_sklearn`) passed.

Earlier runs:

- 2026-11-21, commit f6cc6fb: 412 passed. The test added since checks
  that a simulated dataset's rows record the recall and precision of its
  informative features (commit cbf8763).
- 2026-11-20, commit f67b5c0: 410 passed. The two tests added after it
  check that the runner skips a cell when PyTorch refuses memory under
  the cap, and that it still stops on any other error.
