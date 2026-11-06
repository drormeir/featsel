"""
scikit-learn's LinearSVC objective, solved exactly by Newton's method, for SVM-RFE.

LinearSVC with its defaults minimizes, per class against the rest,

    0.5 |w|^2 + C sum_i max(0, 1 - y_i (w . x_i + b))^2,   y_i in {-1, +1},

with the intercept b penalized like a weight on a constant feature of 1
(liblinear's intercept_scaling=1). With far fewer samples than features the
problem is solved in the samples' space: w = sum_i gamma_i x_i, through the
kernel K = X X^T + 1, an n x n matrix.

The solver is Newton's method on the set S of samples inside the margin
(Chapelle 2007, "Training a support vector machine in the primal"). For a
fixed S the objective is quadratic, and its minimizer solves

    (K_SS + I / (2C)) gamma_S = y_S,   gamma = 0 outside S.

A line search keeps each step a descent; the method stops when S no longer
changes after a full step, which is the exact minimum, or after a step too
small to change anything. liblinear stops at a tolerance instead (1e-4 by
default), so the two agree only as far as that tolerance allows.

Everything runs in float64: in float32 the kernel's rounding alone changed
which features RFE kept. The two products with X, the kernel and the
weights, are the bulk of the work. They run wherever X is: NumPy on the CPU,
or a PyTorch tensor on a GPU. The Newton steps work on n x n matrices,
where a CPU beats this project's GPU in float64, so they always run in
NumPy and SciPy.

On the CPU every product goes through SciPy's BLAS, not NumPy's. The two
packages ship separate OpenBLAS libraries, each with its own threads, and
after a NumPy product its idle threads keep spinning: a SciPy Cholesky
right after one ran six times slower.
"""

import warnings

import numpy as np
from scipy.linalg import blas, cho_factor, cho_solve
from sklearn.exceptions import ConvergenceWarning

# Bisection steps of the line search; 2^-50 of the step is far below float64
# resolution of the objective.
_LINE_SEARCH_STEPS = 50


class Elimination:
    """
    The squared LinearSVC weights at every round of SVM-RFE, for one dataset.

    Rounds only remove features, so the kernel is kept between rounds and
    the removed features' share subtracted, a fraction of recomputing it.
    Each round's Newton method starts from the previous round's set S,
    which changes little between rounds.

    X is float64, shape (n_samples, n_features): a NumPy array, or a PyTorch
    tensor on any device. y is a NumPy array of labels and classes its
    sorted unique values. Two classes give one problem, the second class
    against the first; more give one per class against the rest, as
    LinearSVC's one-vs-rest.
    """

    def __init__(self, X, y, classes, C, max_iter):
        self.X = X
        self.C = C
        self.max_iter = max_iter
        self.K = _gram(X) + 1.0
        self.kept = np.ones(X.shape[1], dtype=bool)
        self.signs = [(y == label) * 2.0 - 1.0
                      for label in (classes[1:] if len(classes) == 2 else classes)]
        self.inside = [None] * len(self.signs)

    def squared_weights(self, features):
        """The squared weights of `features`, sorted indices of the survivors, summed over problems."""
        now = np.zeros_like(self.kept)
        now[features] = True
        dropped = np.flatnonzero(self.kept & ~now)
        if len(dropped):
            self.K -= _gram(self.X[:, dropped])
            self.kept = now
        total = np.zeros(len(features))
        for i, signs in enumerate(self.signs):
            gamma, self.inside[i] = dual_coefficients(self.K, signs, self.C, self.max_iter,
                                                      self.inside[i])
            # One product with all of X: cheaper than copying out the survivors.
            total += _transposed_times(self.X, gamma)[features] ** 2
        return total


def _gram(X):
    """X X^T as a NumPy array, X a NumPy array or a tensor on any device."""
    if not isinstance(X, np.ndarray):
        return (X @ X.T).cpu().numpy()
    # X.T is column-major, as BLAS wants, without a copy. syrk fills one
    # triangle, at half the work of a full product.
    upper = blas.dsyrk(1.0, np.ascontiguousarray(X).T, trans=1)
    return upper + np.triu(upper, 1).T


def _transposed_times(X, v):
    """X^T v as a NumPy array, X a NumPy array or a tensor on any device."""
    if not isinstance(X, np.ndarray):
        return (X.T @ X.new_tensor(v)).cpu().numpy()
    return blas.dgemv(1.0, X.T, v)


def dual_coefficients(K, y, C, max_iter, inside=None):
    """
    gamma, the minimizer's coefficients over the samples, by Newton's method,
    and the final set S as a mask. `inside` is a starting S; all samples if
    None, as at w = 0.
    """
    gamma = np.zeros(len(y))
    f = np.zeros(len(y))
    if inside is None:
        inside = np.ones(len(y), dtype=bool)
    ridge = 1.0 / (2.0 * C)
    for _ in range(max_iter):
        S = np.flatnonzero(inside)
        A = K[np.ix_(S, S)]
        A[np.diag_indices_from(A)] += ridge
        target = np.zeros(len(y))
        target[S] = cho_solve(cho_factor(A, overwrite_a=True, check_finite=False), y[S],
                              check_finite=False)

        step = target - gamma
        K_step = blas.dsymv(1.0, K.T, step)   # K is symmetric; K.T is column-major
        t = _line_search(gamma, step, f, K_step, y, C)
        gamma = gamma + t * step
        f = f + t * K_step
        now_inside = y * f < 1
        # Done when S holds after a full step, which is the exact minimum,
        # or after a step too small to matter: at the minimum, rounding can
        # tilt the line search's slope and stop it short of 1.
        negligible = np.abs(t * step).max() <= 1e-9 * np.abs(gamma).max()
        if np.array_equal(now_inside, inside) and (t == 1.0 or negligible):
            return gamma, inside
        inside = now_inside
    warnings.warn(f"The Newton linear SVM did not converge in {max_iter} steps",
                  ConvergenceWarning, stacklevel=3)
    return gamma, inside


def _line_search(gamma, step, f, K_step, y, C):
    """
    The step length in [0, 1] that minimizes the objective along `step`.

    The objective is convex and piecewise quadratic along the line, so its
    derivative is increasing: a full step is taken when the derivative at 1
    is not positive, else the derivative's root is found by bisection.
    """
    gamma_K_step = gamma @ K_step
    step_K_step = step @ K_step

    def slope(t):
        margin = 1 - y * (f + t * K_step)
        violated = margin > 0
        return (gamma_K_step + t * step_K_step
                - 2 * C * (margin[violated] * y[violated] * K_step[violated]).sum())

    if slope(1.0) <= 0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(_LINE_SEARCH_STEPS):
        middle = (low + high) / 2
        if slope(middle) > 0:
            high = middle
        else:
            low = middle
    return low
