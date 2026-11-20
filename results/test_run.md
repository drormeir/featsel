# Test run, 2026-11-21

Command, from the repository root:

    .venv/bin/python -m pytest -q -p no:cacheprovider

Code: commit f6cc6fb. The run started before that commit, on a working
tree with the same `featsel/`, `featsel_datasets/` and `tests/`.

Machine: Intel i7-9750H, 16 GB, NVIDIA GTX 1650 (4 GB), so the GPU tests
ran.

Result, the last line of pytest's output:

    412 passed, 5 warnings in 1409.31s (0:23:29)

No test failed or was skipped. The 11 speed tests
(`test_not_slower_than_sklearn`) passed.

An earlier run, on 2026-11-20 at commit f67b5c0, gave 410 passed. The two
tests added since check that the runner skips a cell when PyTorch refuses
memory under the cap, and that it still stops on any other error.
