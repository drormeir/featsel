"""
Speed and reproducibility of each selector, against its original.

Each selector has variants: the ways it can run. The first is the original,
the method as published or as scikit-learn implements it; the rest are this
package's versions of it: other solvers, threads or a GPU. benchmark()
fits every variant on a growing number of genes and records, per variant
and size:

- the median time of `repeats` fits (2 by default), after one untimed fit
  (CUDA starts then);
- whether every fit chose the same features (reproducible);
- the Jaccard index of its features against the original's (1 is the same).

Every fit is by a fresh, unfitted selector, so nothing carries over from one
fit to the next. A variant whose first fit takes longer than max_seconds is
fit twice only, and its larger sizes are skipped.

The data is synthetic (make_classification), shaped like SCAN-B: about 3,000
patients, up to 30,000 genes, one class in five positive.

The notebook featsel/selectors/<name>/demos/speed.ipynb runs this for one
selector; scripts/benchmark_selectors.py runs it for all, on SCAN-B.
"""

import resource
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFdr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    f_classif,
    mutual_info_classif,
)
from sklearn.feature_selection import VarianceThreshold as SKLearnVarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from featsel.selectors import FeatureSelector

SEED = 42
N_GENES = (1000, 3000, 10000, 30000)

# Each selector as the experiment configs set it, or with its defaults where
# no config uses it. n_features is 50 where it takes one.
PARAMS = {
    'VarianceThreshold': {'threshold': 0.5},
    'ANOVAFSelector': {'n_features': 50},
    'FDRSelector': {'alpha': 0.05},
    'FWESelector': {'alpha': 0.05},
    'PercentileSelector': {'percentile': 1},
    'MannWhitneySelector': {'n_features': 50},
    'HigherCriticismSelector': {'variant': '2008'},
    'CorrelationSelector': {'n_features': 50},
    'MutualInfoSelector': {'n_features': 50},
    'LassoSelector': {'C': 1.0},
    'TreeImportanceSelector': {'n_features': 50, 'n_estimators': 200},
    'RFESelector': {'n_features': 50, 'C': 0.01},
    'RFECVSelector': {'C': 0.01, 'cv': 5},
    'SequentialSelector': {'n_features': 10, 'cv': 5},
    'RandomSelector': {'n_features': 50},
}


def _svm(C):
    return LinearSVC(C=C, max_iter=1000, random_state=SEED)


def _pandas_correlation(**params):
    """CorrelationSelector finding redundant pairs from the full pandas matrix, as it once did."""
    selector = FeatureSelector.lookup('CorrelationSelector')(**params, device='cpu')

    def pairs(X):
        corr = X.corr(method=selector.method).abs().to_numpy()
        return zip(*np.nonzero(np.triu(corr > selector.inter_feature_threshold, k=1)))

    selector._redundant_pairs = pairs
    return selector


# selector -> the original: a function of the selector's PARAMS that builds it
# unfitted. Mann-Whitney and HC have no scikit-learn version: their original
# is this package's NumPy path, on the CPU.
ORIGINALS = {
    'VarianceThreshold': lambda threshold: SKLearnVarianceThreshold(threshold),
    'ANOVAFSelector': lambda n_features: SelectKBest(f_classif, k=n_features),
    'FDRSelector': lambda alpha: SelectFdr(f_classif, alpha=alpha),
    'FWESelector': lambda alpha: SelectFwe(f_classif, alpha=alpha),
    'PercentileSelector': lambda percentile: SelectPercentile(f_classif, percentile=percentile),
    'MutualInfoSelector': lambda n_features: SelectKBest(
        partial(mutual_info_classif, n_neighbors=3, random_state=SEED), k=n_features),
    'CorrelationSelector': _pandas_correlation,
    # The threshold, the smallest positive float, keeps the non-zero
    # coefficients, as LassoSelector does.
    'LassoSelector': lambda C: SelectFromModel(
        LogisticRegression(l1_ratio=1, C=C, solver='liblinear', max_iter=1000,
                           random_state=SEED),
        threshold=np.finfo(float).tiny),
    'TreeImportanceSelector': lambda n_features, n_estimators: SelectFromModel(
        RandomForestClassifier(n_estimators=n_estimators, random_state=SEED),
        max_features=n_features, threshold=-np.inf),
    'RFESelector': lambda n_features, C: RFE(_svm(C), n_features_to_select=n_features,
                                             step=0.1),
    'RFECVSelector': lambda C, cv: RFECV(_svm(C), step=0.1, cv=cv),
    'SequentialSelector': lambda n_features, cv: SequentialFeatureSelector(
        LogisticRegression(C=1.0, max_iter=1000), n_features_to_select=n_features, cv=cv),
    'MannWhitneySelector': lambda n_features: FeatureSelector.create(
        'MannWhitneySelector', n_features=n_features, device='cpu'),
    'HigherCriticismSelector': lambda variant: FeatureSelector.create(
        'HigherCriticismSelector', variant=variant, device='cpu'),
}

# selector -> this package's variants: label -> parameters added to PARAMS.
# A selector not listed has one, 'featsel', with PARAMS as they are.
_CPU_AND_GPU = {'featsel, CPU': {'device': 'cpu'}, 'featsel, GPU': {'device': 'cuda'}}

VARIANTS = {
    'VarianceThreshold': _CPU_AND_GPU,
    'ANOVAFSelector': _CPU_AND_GPU,
    'FDRSelector': _CPU_AND_GPU,
    'FWESelector': _CPU_AND_GPU,
    'PercentileSelector': _CPU_AND_GPU,
    'CorrelationSelector': _CPU_AND_GPU,
    # 'featsel' keeps the label of the CPU series measured before the GPU path.
    'MutualInfoSelector': {'featsel': {'device': 'cpu'}, 'featsel, GPU': {'device': 'cuda'}},
    'MannWhitneySelector': {'GPU': {'device': 'cuda'}},
    'HigherCriticismSelector': {'GPU': {'device': 'cuda'}},
    # PPDNA on the CPU is about five times slower than liblinear; it pays
    # only on a GPU.
    'LassoSelector': {'featsel PPDNA, GPU': {'solver': 'ppdna', 'device': 'cuda'}},
    'TreeImportanceSelector': {'featsel, all threads': {'n_jobs': -1}},
    'RFESelector': {'featsel Newton, CPU': {'solver': 'newton'},
                    'featsel Newton, GPU': {'solver': 'newton', 'device': 'cuda'}},
    'RFECVSelector': {'featsel Newton, CPU': {'solver': 'newton'},
                      'featsel Newton, GPU': {'solver': 'newton', 'device': 'cuda'}},
    'SequentialSelector': {'featsel Newton, CPU': {'solver': 'newton'},
                           'featsel Newton, GPU': {'solver': 'newton', 'device': 'cuda'}},
}

ORIGINAL_LABEL = {
    'MannWhitneySelector': 'NumPy, CPU',
    'HigherCriticismSelector': 'NumPy, CPU',
    'CorrelationSelector': 'pandas full matrix',
}


def variants(name, gpu=None):
    """
    The selector's variants to run here, label -> function building one
    unfitted. The original comes first. GPU variants are left out without a
    usable GPU; gpu=None checks for one.
    """
    if gpu is None:
        gpu = _gpu_usable()
    params = PARAMS[name]
    built = {}
    if name in ORIGINALS:
        built[ORIGINAL_LABEL.get(name, 'scikit-learn')] = partial(ORIGINALS[name], **params)
    for label, extra in VARIANTS.get(name, {'featsel': {}}).items():
        if gpu or extra.get('device') != 'cuda':
            built[label] = partial(FeatureSelector.create, name, **params, **extra)
    return built


def synthetic(n_samples=3000, n_genes=30000, seed=SEED):
    """
    Random genes and a binary target, one patient in five positive, shaped
    like SCAN-B. 50 genes are informative, scattered among the rest. float32,
    as the runner holds data.
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_genes, n_informative=50,
                               n_redundant=0, weights=[0.8], flip_y=0.01,
                               random_state=seed)
    return X.astype(np.float32), y


def benchmark(name, X, y, n_genes=N_GENES, repeats=2, max_seconds=60.0, gpu=None):
    """
    One row per variant and gene count: times, reproducibility and agreement.

    Two timed fits are enough to see a gap of several times, such as an
    original against a GPU version. A gap of a few percent needs many more,
    and the two versions fit alternately: drift in the machine's speed
    (power, heat) shifts all later fits, and more repeats do not remove it.

    The genes at each size are the first n_genes columns of X, so each
    larger set contains the smaller ones.
    """
    rows, reference = [], {}
    for label, build in variants(name, gpu).items():
        for n in sorted(n_genes):
            X_n = np.ascontiguousarray(X[:, :n])
            seconds, supports, device = _fits(build, X_n, y, repeats, max_seconds)
            reference.setdefault(n, supports[0])
            rows.append({
                'variant': label, 'device': device, 'n_genes': n, 'n_patients': len(y),
                'fits': len(supports), 'median_s': float(np.median(seconds)),
                'min_s': min(seconds), 'max_s': max(seconds),
                'n_selected': int(supports[0].sum()),
                'reproducible': all(np.array_equal(supports[0], s) for s in supports[1:]),
                'jaccard_vs_original': _jaccard(supports[0], reference[n]),
            })
            if rows[-1]['median_s'] > max_seconds:
                break
    return pd.DataFrame(rows)


def _fits(build, X, y, repeats, max_seconds):
    """
    Seconds of the timed fits, the support of every fit, and the device.
    One untimed fit first; if it took longer than max_seconds, one timed
    fit follows instead of `repeats`.
    """
    def fit():
        selector = build()
        start = time.perf_counter()
        selector.fit(X, y)
        return time.perf_counter() - start, selector

    warm_up, selector = fit()
    supports, seconds = [selector.get_support()], []
    for _ in range(1 if warm_up > max_seconds else repeats):
        elapsed, selector = fit()
        seconds.append(elapsed)
        supports.append(selector.get_support())
    return seconds, supports, getattr(selector, 'device_', 'cpu')


def _jaccard(a, b):
    either = (a | b).sum()
    return float((a & b).sum() / either) if either else float('nan')


# One color per role, the same in every notebook: the original gray, this
# package's CPU versions blue, its GPU versions orange. Each line has its own
# marker and an end label too, so none is told apart by color alone.
ORIGINAL, CPU, GPU = '#5f6368', '#2a78d6', '#e8710a'
_MARKERS = 'os^D'


def plot(table, title):
    """Median fit time against gene count, log-log, one line per variant."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (label, part) in enumerate(table.groupby('variant', sort=False)):
        color = ORIGINAL if i == 0 else GPU if (part['device'] == 'cuda').all() else CPU
        marker = _MARKERS[i]
        ax.plot(part['n_genes'], part['median_s'], color=color, marker=marker,
                linewidth=2, markersize=7, markeredgecolor='white', label=label)
        last = part.iloc[-1]
        ax.annotate(label, (last['n_genes'], last['median_s']), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=8, color='#1f2328')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Genes', color='#1f2328')
    ax.set_ylabel('Seconds per fit (median)', color='#1f2328')
    ax.set_title(title, loc='left', color='#1f2328')
    ax.grid(True, which='major', color='#e3e5e8', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors='#5f6368')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_xlim(right=ax.get_xlim()[1] * 3)   # room for the end labels
    fig.tight_layout()
    return fig


def cap_memory(fraction=0.8, gb=None):
    """
    Cap this process's data memory at gb, or else at a fraction of the RAM
    free now, so a fit that would exhaust the machine raises MemoryError
    instead. Returns the cap in GB.
    """
    if gb is not None:
        cap = int(gb * 1024 ** 3)
    else:
        with open('/proc/meminfo') as handle:
            available = next(int(line.split()[1]) * 1024 for line in handle
                             if line.startswith('MemAvailable:'))
        cap = int(fraction * available)
    resource.setrlimit(resource.RLIMIT_DATA, (cap, resource.getrlimit(resource.RLIMIT_DATA)[1]))
    return cap / 1024 ** 3


def _gpu_usable():
    from featsel.selectors import gpu
    return gpu.cuda_usable()
