"""
Checks that compare a selector with its scikit-learn equivalent.

Used by the tests of each selector that wraps a scikit-learn method, with the
sklearn_reference_data fixture.
"""

import time

import numpy as np

REPEATS = 10
MAX_SLOWDOWN = 1.05


def assert_same_selection(ours, reference, X, y):
    """Both, fitted on X and y, keep exactly the same features."""
    np.testing.assert_array_equal(ours.fit(X, y).get_support(),
                                  reference.fit(X, y).get_support())


def _median_selection_time(selector, X, y):
    """
    Median time to fit and get the support mask.

    scikit-learn builds the mask in get_support(), ours in fit(), so timing
    fit() alone would charge only ours for it.
    """
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        selector.fit(X, y).get_support()
        times.append(time.perf_counter() - start)
    return np.median(times)


def assert_not_slower(ours, reference, X, y):
    """
    Our median selection time over REPEATS runs is at most 5% above scikit-learn's.

    Timed without our own input check, which scikit-learn repeats inside.
    """
    ours.set_params(check_input=False)

    ours_time = _median_selection_time(ours, X, y)
    reference_time = _median_selection_time(reference, X, y)

    assert ours_time <= MAX_SLOWDOWN * reference_time, (
        f"{type(ours).__name__} median selection {ours_time * 1e3:.2f} ms vs "
        f"scikit-learn {reference_time * 1e3:.2f} ms"
    )
