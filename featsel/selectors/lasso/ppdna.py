"""
liblinear's L1-penalized logistic regression, solved by PPDNA on a GPU.

LogisticRegression(l1_ratio=1, solver='liblinear') minimizes

    |w|_1 + C sum_i log(1 + exp(-s_i (x_i . w + b))),   s_i in {-1, +1},

with the intercept b penalized like a weight on a constant feature of 1
(liblinear's intercept_scaling=1). Divided by C, this is

    lam |w|_1 + h(A w),   lam = 1 / C,   h(z) = sum_i log(1 + exp(-s_i z_i)),

where A is X with a column of ones appended and w includes b.

The solver is PPDNA, the dual Newton proximal point algorithm of Liu and
Zhou (2023, arXiv 2310.19353). Each outer step is a proximal point step,

    w_{k+1} = argmin_w  lam |w|_1 + h(A w) + |w - w_k|^2 / (2 sigma),

solved through its dual, a smooth problem in n_samples variables u:

    psi(u) = h*(u) + |p|^2 / (2 sigma) + constant,   p = soft(w_k - sigma A^T u, sigma lam),

then w_{k+1} = p. h* is the logistic loss's conjugate, sum_i t_i log t_i +
(1 - t_i) log(1 - t_i) with t_i = -s_i u_i in (0, 1). psi is minimized by a
semismooth Newton method. Its Newton system involves only the genes active in
p: with r of them and n samples, it is solved through the Woodbury identity
as an r x r system when r < n, else as an n x n one. That keeps every step a
few dense matrix products, which is what a GPU does well.

Differences from the paper, found necessary on SCAN-B-shaped data:

- The intercept is penalized, as liblinear penalizes it: it is a column of A.
- sigma starts at 40 C, the paper's start in this scaling, and grows five-fold
  per outer step up to 1e6 C. Past that, x = w - sigma A^T u loses the digits
  p is made of.
- The Newton matrix is formed and factored in float32, where a consumer GPU
  is many times faster than in float64, and the solution refined in float64
  against the float64 matrix. When float32 cannot reach full accuracy, the
  step is solved in float64.
- The inner loop also stops when full Newton steps no longer shrink the
  gradient: float64's floor, below which more steps only add noise.

The paper's inner stopping test is kept: the subproblem's duality gap,
f_k(p) + psi_k(u) <= min(eps_k^2, eps_k^2 |p - w_k|^2) / (2 sigma), with
eps_k = 9 / k^1.01. The method stops when the relative KKT residual of the
original problem falls below 1e-10, which matches liblinear run to a tight
tolerance, not liblinear's default: that stops at 1e-4 and keeps slightly
different genes.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from .. import gpu

# The starting dual point: t_i = 2e-7, as the paper's u0. Far fewer genes are
# active near it than at t = 1/2, so the first Newton steps are cheap.
_T0 = 2e-7
_SIGMA0, _GROW, _SIGMA_MAX = 40.0, 5.0, 1e6    # times C
_TOL = 1e-10
_NO_GAIN_STEPS = 3      # outer steps without halving the residual: the rounding floor
_MAX_INNER = 100
_REFINE = 10                                   # float64 refinement steps per Newton step
_LINE_SEARCH_STEPS = 60


def l1_logistic(X, signs, C, device, max_outer):
    """
    liblinear's L1 logistic regression of signs (+1 / -1) on X, on device.

    X is a NumPy array or a PyTorch tensor, shape (n_samples, n_features).
    Returns the coefficients (n_features,) and the intercept, as NumPy.
    """
    torch = gpu.import_torch(device)
    if isinstance(X, np.ndarray):
        X = np.require(X, requirements=['W'])   # PyTorch wants a writable array
    X = torch.as_tensor(X, device=device)
    A = torch.cat([X.to(torch.float64), X.new_ones((X.shape[0], 1), dtype=torch.float64)], 1)
    del X
    s = torch.as_tensor(signs, dtype=torch.float64, device=device)
    w = _ppdna(torch, A, A.float(), s, 1.0 / C, C, max_outer).cpu().numpy()
    return w[:-1], float(w[-1])


def _ppdna(torch, A, A32, s, lam, C, max_outer):
    w = A.new_zeros(A.shape[1])
    u = -_T0 * s
    Atu = A.T @ u
    sigma, sigma_max = _SIGMA0 * C, _SIGMA_MAX * C
    best, best_x, no_gain = np.inf, (w, 0.0), 0
    for outer in range(1, max_outer + 1):
        u, Atu = _newton(torch, A, A32, s, lam, sigma, w, u, Atu, 9.0 / outer ** 1.01)
        x = w - sigma * Atu
        w = _soft(torch, x, sigma * lam)
        residual = _kkt_residual(torch, A, s, w, lam)
        if residual < _TOL:
            return _without_rounding_noise(torch, x, sigma * lam)
        # On many genes float64's rounding sets a floor near 1e-9: there the
        # residual wanders instead of falling. Three outer steps without
        # halving the best residual means the floor is reached; the best
        # weights are as good as float64 allows.
        if residual < best / 2:
            best, best_x, no_gain = residual, (x, sigma * lam), 0
        else:
            no_gain += 1
            if no_gain == _NO_GAIN_STEPS:
                return _without_rounding_noise(torch, *best_x)
        sigma = min(sigma * _GROW, sigma_max)
    warnings.warn(f"PPDNA did not converge in {max_outer} outer steps",
                  ConvergenceWarning, stacklevel=4)
    return _without_rounding_noise(torch, *best_x)


def _without_rounding_noise(torch, x, threshold):
    """
    The final weights, soft(x, threshold), with weights below their rounding
    error set to 0.

    A weight is |x| - threshold, the difference of two numbers about
    threshold in size, so float64 rounds it by about threshold * eps. A weight
    smaller than that is noise: it counts as 0, and its gene is not selected.
    The solver's own steps use the plain soft threshold.
    """
    w_temp = x.abs() - threshold
    w = torch.sign(x) * w_temp
    w[w_temp < threshold * np.finfo(np.float64).eps] = 0
    return w


def _newton(torch, A, A32, s, lam, sigma, w, u, Atu, eps):
    """Minimize psi from u by semismooth Newton; return u and A^T u."""
    w_norm2 = float(w @ w)
    stalled, previous, alpha = 0, None, 0.0
    for _ in range(_MAX_INNER):
        p = _soft(torch, w - sigma * Atu, sigma * lam)
        active = torch.nonzero(p).flatten()
        t = -s * u
        Ap = A[:, active] @ p[active]
        grad = s * torch.log((1 - t) / t) - Ap
        p_norm2 = float(p @ p)
        # The subproblem's duality gap, the paper's stopping test.
        primal = (float(torch.nn.functional.softplus(-s * Ap).sum())
                  + lam * float(p.abs().sum()) + float((p - w) @ (p - w)) / (2 * sigma))
        dual = float(_entropy(torch, t)) + (p_norm2 - w_norm2) / (2 * sigma)
        if primal + dual <= eps ** 2 * min(1.0, float((p - w) @ (p - w))) / (2 * sigma):
            break
        gnorm = float(grad.norm())
        if (alpha == 1.0 and gnorm >= 0.5 * previous
                and gnorm <= 1e-8 * (1 + p_norm2 ** 0.5)):
            stalled += 1
            if stalled == 2:
                break
        else:
            stalled = 0
        previous = gnorm

        d = _direction(torch, A, A32, active, t * (1 - t), -grad, sigma)
        Atd = A.T @ d
        psi0 = float(_entropy(torch, t)) + p_norm2 / (2 * sigma)
        slope = float(grad @ d)
        alpha = 1.0
        for _ in range(_LINE_SEARCH_STEPS):
            t_new = -s * (u + alpha * d)
            if bool(((t_new > 0) & (t_new < 1)).all()):
                p_new = _soft(torch, w - sigma * (Atu + alpha * Atd), sigma * lam)
                psi = float(_entropy(torch, t_new)) + float(p_new @ p_new) / (2 * sigma)
                if psi <= psi0 + 0.01 * alpha * slope:
                    break
            alpha *= 0.6
        else:
            break        # no decrease left to find at float64's resolution
        u = u + alpha * d
        Atu = Atu + alpha * Atd
    return u, Atu


def _direction(torch, A, A32, active, Dinv, rhs, sigma):
    """
    Solve (D + sigma A_J A_J^T) d = rhs, D = diag(1 / Dinv), J the active genes.

    Scaled by L = sqrt(Dinv) it is (I + sigma Ahat Ahat^T) z = L rhs with
    Ahat = L A_J and d = L z, a matrix at least I, so well conditioned.
    """
    if len(active) == 0:
        return Dinv * rhs
    L = Dinv.sqrt()
    Ahat = L[:, None] * A[:, active]
    Ahat32 = L[:, None].float() * A32[:, active]
    b = L * rhs
    if len(active) >= A.shape[0]:
        def form(Ah):
            N = sigma * (Ah @ Ah.T)
            N.diagonal().add_(1.0)
            return N
        return L * _solve(torch, form, Ahat, Ahat32,
                          lambda z: z + sigma * (Ahat @ (Ahat.T @ z)), b)

    # Woodbury: z = b - Ahat v, (Ahat^T Ahat + I / sigma) v = Ahat^T b.
    def form(Ah):
        M = Ah.T @ Ah
        M.diagonal().add_(1 / sigma)
        return M
    v = _solve(torch, form, Ahat, Ahat32, lambda v: Ahat.T @ (Ahat @ v) + v / sigma, Ahat.T @ b)
    return L * (b - Ahat @ v)


def _solve(torch, form, Ahat, Ahat32, apply, b):
    """
    Solve M x = b, M = form(Ahat): factored in float32 and refined in float64
    with apply(x) = M x, or in float64 when float32 cannot reach full accuracy.
    """
    factor, info = torch.linalg.cholesky_ex(form(Ahat32))
    if int(info) == 0:
        x = torch.cholesky_solve(b.float()[:, None], factor)[:, 0].double()
        for _ in range(_REFINE):
            r = b - apply(x)
            if float(r.norm()) <= 1e-13 * float(b.norm()):
                return x
            x = x + torch.cholesky_solve(r.float()[:, None], factor)[:, 0].double()
    factor = torch.linalg.cholesky(form(Ahat))
    return torch.cholesky_solve(b[:, None], factor)[:, 0]


def _soft(torch, x, threshold):
    return torch.sign(x) * torch.clamp(x.abs() - threshold, min=0)


def _entropy(torch, t):
    return (t * torch.log(t) + (1 - t) * torch.log1p(-t)).sum()


def _kkt_residual(torch, A, s, w, lam):
    """|w - prox(w - gradient)| relative to |w| and |gradient|: 0 at the minimum."""
    g = A.T @ (-s * torch.sigmoid(-s * (A @ w)))
    return float((w - _soft(torch, w - g, lam)).norm() / (1 + w.norm() + g.norm()))
