# Test run, 2026-11-20

Command, from the repository root:

    .venv/bin/python -m pytest -q -p no:cacheprovider

Code: commit f67b5c0. `featsel/`, `featsel_datasets/` and `tests/` have not
changed since then (to commit 46674fc).

Machine: Intel i7-9750H, 16 GB, NVIDIA GTX 1650 (4 GB), so the GPU tests
ran. Other programs ran on the laptop at the same time.

Result, the last line of pytest's output:

    410 passed, 5 warnings in 1407.45s (0:23:27)

No test failed or was skipped. The 11 speed tests
(`test_not_slower_than_sklearn`) passed under that load.
