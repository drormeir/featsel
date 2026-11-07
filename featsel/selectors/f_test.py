"""
The F-test of every feature, as scikit-learn's f_classif and f_regression,
on the CPU or a GPU.

On the CPU it is scikit-learn's own function. On a GPU only the sums over
samples run there, in float64, in batches of features that fit the GPU's
memory; the p-values follow in NumPy from the same formulas. The GPU's sums
round differently, so a feature whose statistic ties another's to the last
digits can rank on the other side of it.
"""

import numpy as np
from scipy import special, stats
from sklearn.feature_selection import f_classif, f_regression

from . import gpu

# Peak bytes per sample of one feature on the GPU: the float64 copy and its square.
_BYTES_PER_VALUE = 16


def f_test(X, y, task='classification', device='auto'):
    """
    F statistic and p-value of each feature, and the device used.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    task : {'classification', 'regression'}
        f_classif's one-way ANOVA, or f_regression's test of a linear fit.
    device : str, default='auto'
        As gpu.run: 'auto', 'cpu', or a PyTorch device such as 'cuda'.

    Returns
    -------
    scores, pvalues : np.ndarray
    device : str
    """
    if task not in ('classification', 'regression'):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    if task == 'classification':
        cpu, on_device = f_classif, _f_classif_torch
    else:
        cpu, on_device = f_regression, _f_regression_torch
    (scores, pvalues), device = gpu.run(device, X, lambda X: cpu(X, y),
                                        lambda X, device: on_device(X, y, device))
    return scores, pvalues, device


def _f_classif_torch(X, y, device):
    """f_classif's one-way ANOVA, its sums on device."""
    torch = gpu.import_torch(device)
    classes, codes = np.unique(y, return_inverse=True)
    counts = np.bincount(codes)
    n_samples, n_classes = len(y), len(classes)
    one_hot = torch.as_tensor(np.eye(n_classes)[codes], device=device)
    parts = []
    for block in gpu.float64_columns(torch, X, device, _BYTES_PER_VALUE):
        class_sums = one_hot.T @ block                          # (n_classes, batch)
        squares = (block * block).sum(0)
        parts.append(torch.vstack([class_sums, squares[None]]).cpu().numpy())
    sums = np.hstack(parts)
    class_sums, ss_alldata = sums[:-1], sums[-1]
    square_of_sums_alldata = class_sums.sum(0) ** 2
    sstot = ss_alldata - square_of_sums_alldata / n_samples
    ssbn = (class_sums ** 2 / counts[:, None]).sum(0) - square_of_sums_alldata / n_samples
    sswn = sstot - ssbn
    dfbn, dfwn = n_classes - 1, n_samples - n_classes
    with np.errstate(divide='ignore', invalid='ignore'):   # a constant feature: 0 / 0
        f = (ssbn / dfbn) / (sswn / dfwn)
    return f, special.fdtrc(dfbn, dfwn, f)


def _f_regression_torch(X, y, device):
    """f_regression's test, the correlations computed on device."""
    torch = gpu.import_torch(device)
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    n_samples = len(y)
    y_t = torch.as_tensor(y, device=device)
    parts = []
    for block in gpu.float64_columns(torch, X, device, _BYTES_PER_VALUE):
        means = block.mean(0)
        norms = torch.sqrt((block * block).sum(0) - n_samples * means ** 2)
        parts.append((y_t @ block / norms).cpu().numpy())
    corr = np.hstack(parts) / np.linalg.norm(y)
    corr[np.isnan(corr)] = 0.0                                # a constant feature
    dof = n_samples - 2
    corr_squared = corr ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        f = corr_squared / (1 - corr_squared) * dof
        pvalues = stats.f.sf(f, 1, dof)
    f[np.isinf(f)] = np.finfo(f.dtype).max                    # a perfect correlation
    undefined = np.isnan(f)                                   # a constant feature
    f[undefined], pvalues[undefined] = 0.0, 1.0
    return f, pvalues
