import numpy as np

from featsel.strength import loo_confusion

CS = [0.01, 1.0, 100.0]


def _data(n=40, k=5, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, k))
    X[:, 0] += 10 * y
    return X, y


def test_counts_alpha_fraction_of_rows():
    X, y = _data()
    assert loo_confusion(X, y, alpha=0.1, Cs=CS).sum() == 4


def test_alpha_zero_visits_one_row():
    X, y = _data()
    assert loo_confusion(X, y, alpha=0.0, Cs=CS).sum() == 1


def test_alpha_one_visits_every_row():
    X, y = _data(n=20)
    assert loo_confusion(X, y, alpha=1.0, Cs=CS).sum() == 20


def test_separable_data_is_classified_correctly():
    X, y = _data()
    cm = loo_confusion(X, y, alpha=0.5, Cs=CS)
    assert np.trace(cm) == cm.sum()


def test_same_seed_same_result():
    X, y = _data()
    np.testing.assert_array_equal(
        loo_confusion(X, y, alpha=0.2, random_state=3, Cs=CS),
        loo_confusion(X, y, alpha=0.2, random_state=3, Cs=CS),
    )
