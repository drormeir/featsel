"""
Forward sequential selection with many logistic regressions solved at once.

scikit-learn's SequentialFeatureSelector, with LogisticRegression and its
defaults, adds at each step the candidate feature whose cross-validated
accuracy is highest. Every candidate in every fold is one small problem:
the features chosen so far, the candidate, and an intercept. Here all the
candidates of a fold are solved together, as one batch, by Newton's method
in PyTorch, on the CPU or a GPU.

LogisticRegression minimizes, for labels y in {0, 1},

    0.5 |w|^2 + C sum_i (log(1 + exp(eta_i)) - y_i eta_i),   eta = X w + b,

with the intercept b not penalized. The problem is strictly convex, so
Newton's method with a backtracking line search reaches its minimum; it
stops when every problem's step is below 1e-10. lbfgs stops at a tolerance
(1e-4), so the two agree to that tolerance.

Each step's score is the mean of the folds' accuracies, as cross_val_score
gives it, and the first best candidate in index order is taken, as
scikit-learn's max over candidates does.

All in float64: accuracy is discrete, and a rounding difference near the
decision boundary flips a prediction and can change which candidate wins.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

_STEP_TOL = 1e-10
_HALVINGS = 30

# Bytes per float64 value. A batch holds about eight (batch, n) matrices at
# its peak: the candidate columns, eta, the weights and their products.
_BYTES_PER_VALUE = 8


def forward_selection(torch, X, y, n_select, folds, C, max_iter, gpu):
    """
    The indices of the n_select features forward selection adds, in order.

    X is a float64 tensor (n_samples, n_features) on the device; y a NumPy
    array of 0 and 1; folds a list of (train, test) index arrays.

    Each candidate's fit starts from its solution at the previous step, with
    a weight of 0 for the feature added since: the two problems differ by
    that one column, so Newton's method needs few steps.
    """
    n_features = X.shape[1]
    y_t = torch.as_tensor(y, dtype=X.dtype, device=X.device)
    # Per fold, the last solution of every feature as a candidate:
    # [weights of the chosen features, candidate weight, intercept].
    warm = [X.new_zeros(n_features, 2) for _ in folds]
    selected = []
    for _ in range(n_select):
        candidates = np.setdiff1d(np.arange(n_features), selected)
        fold_scores = np.empty((len(folds), len(candidates)))
        for f, (train, test) in enumerate(folds):
            fold_scores[f] = _fold_accuracies(torch, X, y_t, train, test, selected,
                                              candidates, warm[f], C, max_iter, gpu)
        # As cross_val_score: the mean over folds, then the first best.
        winner = int(candidates[np.argmax(fold_scores.mean(axis=0))])
        selected.append(winner)
        # The winner joins the chosen features, after the earlier ones and
        # before the candidate, with a starting weight of 0.
        j = len(selected) - 1
        warm = [torch.cat([theta[:, :j], theta.new_zeros(n_features, 1), theta[:, j:]], dim=1)
                for theta in warm]
    return selected


def _fold_accuracies(torch, X, y, train, test, selected, candidates, warm, C, max_iter, gpu):
    """
    Test accuracy of every candidate added to `selected`, fit on one fold's
    train part. Each fit starts from, and leaves its solution in, warm.
    """
    device = X.device
    train_t = torch.as_tensor(train, device=device)
    test_t = torch.as_tensor(test, device=device)
    X_train, X_test = X[train_t], X[test_t]
    y_train, y_test = y[train_t], y[test_t]
    S_train, S_test = X_train[:, selected], X_test[:, selected]
    # Products of every pair of chosen features, per sample: the Hessian's
    # chosen-by-chosen block for all candidates is one product with these.
    pairs = (S_train[:, :, None] * S_train[:, None, :]).reshape(len(train), -1)
    batch = gpu.batch_size(torch, device, 8 * _BYTES_PER_VALUE * len(train), len(candidates))
    accuracies = np.empty(len(candidates))
    for c0 in range(0, len(candidates), batch):
        chunk = torch.as_tensor(candidates[c0:c0 + batch], device=device)
        theta = _fit(torch, S_train, pairs, X_train[:, chunk].T, y_train, C, max_iter,
                     warm[chunk])
        warm[chunk] = theta
        eta = _linear(S_test, X_test[:, chunk].T, theta)
        correct = ((eta > 0).to(y_test.dtype) == y_test).sum(dim=1)
        accuracies[c0:c0 + batch] = correct.cpu().numpy() / len(test)
    return accuracies


def _linear(S, cand, theta):
    """eta = [S, candidate, 1] theta for every candidate: shape (batch, n)."""
    j = S.shape[1]
    return theta[:, :j] @ S.T + theta[:, j:j + 1] * cand + theta[:, j + 1:]


def _fit(torch, S, pairs, cand, y, C, max_iter, theta):
    """
    Newton's method on a batch of logistic regressions, one per candidate.

    Each problem's features are the shared chosen features S (n, j), its
    own candidate column, a row of cand (batch, n), and an intercept; theta
    is (batch, j + 2) in that order. The design matrices are never built:
    every block of the gradient and the Hessian is a product of a
    (batch, n) matrix with S or pairs, or a sum over samples.
    """
    j = S.shape[1]
    d = j + 2
    penalized = S.new_ones(d)
    penalized[-1] = 0.0                       # the intercept is not penalized
    theta = theta.clone()
    eta = _linear(S, cand, theta)
    objective = _objective(torch, eta, y, theta, C, penalized)
    active = torch.ones(len(theta), dtype=torch.bool, device=S.device)
    for _ in range(max_iter):
        p = torch.sigmoid(eta)
        r = C * (p - y)
        gradient = torch.cat([r @ S, (r * cand).sum(1, keepdim=True), r.sum(1, keepdim=True)],
                             dim=1) + penalized * theta
        w = C * p * (1 - p)
        w_cand = w * cand
        hessian = S.new_empty(len(theta), d, d)
        hessian[:, :j, :j] = (w @ pairs).view(len(theta), j, j)
        hessian[:, :j, j] = w_cand @ S
        hessian[:, :j, j + 1] = w @ S
        hessian[:, j, j] = (w_cand * cand).sum(1)
        hessian[:, j, j + 1] = w_cand.sum(1)
        hessian[:, j + 1, j + 1] = w.sum(1)
        hessian[:, j:, :j] = hessian[:, :j, j:].transpose(1, 2)
        hessian[:, j + 1, j] = hessian[:, j, j + 1]
        hessian.diagonal(dim1=1, dim2=2).add_(penalized)
        step = torch.linalg.solve(hessian, gradient) * active[:, None]
        eta_step = _linear(S, cand, step)

        # Backtracking, rarely needed: halve the step wherever the objective
        # rose by more than rounding. eta is linear in the step, so no new
        # product is needed.
        t = S.new_ones(len(theta))
        slack = 1e-12 * (1 + objective.abs())
        for _ in range(_HALVINGS):
            new_theta = theta - t[:, None] * step
            new_eta = eta - t[:, None] * eta_step
            new_objective = _objective(torch, new_eta, y, new_theta, C, penalized)
            worse = new_objective > objective + slack
            if not worse.any():
                break
            t = torch.where(worse, t / 2, t)
        # A step that never lowered the objective is not taken: that
        # problem is at its minimum to rounding.
        keep = worse[:, None]
        theta = torch.where(keep, theta, new_theta)
        eta = torch.where(keep, eta, new_eta)
        objective = torch.where(worse, objective, new_objective)
        small = (t[:, None] * step).abs().amax(dim=1) <= _STEP_TOL * (1 + theta.abs().amax(dim=1))
        active &= ~(small | worse)
        if not active.any():
            return theta
    warnings.warn(f"Batched logistic regression did not converge in {max_iter} Newton steps",
                  ConvergenceWarning, stacklevel=4)
    return theta


def _objective(torch, eta, y, theta, C, penalized):
    """0.5 |w|^2 + C * log-loss per problem, from its linear predictor eta."""
    loss = torch.nn.functional.softplus(eta) - y * eta
    return 0.5 * (penalized * theta ** 2).sum(dim=1) + C * loss.sum(dim=1)
