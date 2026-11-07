"""
Time every selector on every device it supports, on one machine, one way.

The series behind any speed claim about a selector. Each selector fits on
the same data at a growing number of genes, once untimed to warm up (CUDA
starts then), then --repeats times timed. A row gives the median and the
range of those fits, the GPU memory the fit peaked at, and how far its
selection is from the selector's first variant, its reference (Jaccard
index, 1 is the same genes). A selector's variants are the ways it can
run: its solvers, devices or thread counts.

Once a variant's fit passes --max-seconds, it is timed once and its larger
sizes are skipped: the series shows where it stops being usable, without
running for hours.

The process's data memory is capped (--max-memory-gb, by default 80% of
the RAM free at the start), so a fit that would exhaust the machine raises
MemoryError and ends the series instead.

The data is SCAN-B, Basal vs rest, the training half of one split at train
fraction 0.5, scaled on that half and held in float32, as the runner holds
it. Genes are a random subset, the same for every selector; each larger
subset contains the smaller ones. --synthetic replaces SCAN-B by random
data of the same shape, to check the script end to end in seconds.

Each row is appended to a CSV as soon as it is measured, so an interrupted
series keeps what it finished. Beside it a JSON file records the arguments,
the machine, the library versions and the git commit. Both go to
results/local/benchmark_selectors/, which git ignores, named by start time.

Usage:
    python scripts/benchmark_selectors.py --synthetic --n-genes 500 2000 --repeats 2
    python scripts/benchmark_selectors.py --selectors RFESelector --n-genes 1000 3000
"""

import argparse
import csv
import resource
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from featsel import environment
from featsel.selectors import FeatureSelector

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'results' / 'local' / 'benchmark_selectors'

# Each selector as the experiment configs set it, or with its defaults where
# no config uses it. n_features is 50 where it takes one.
SELECTORS = {
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

# The ways each selector can run, by label: the parameters added to the
# above. The first is the reference every other variant is compared with.
VARIANTS = {
    'MannWhitneySelector': {'cpu': {'device': 'cpu'}, 'cuda': {'device': 'cuda'}},
    'HigherCriticismSelector': {'cpu': {'device': 'cpu'}, 'cuda': {'device': 'cuda'}},
    'TreeImportanceSelector': {'1 thread': {'n_jobs': 1}, 'all threads': {'n_jobs': -1}},
    'RFESelector': {'liblinear': {},
                    'newton cpu': {'solver': 'newton'},
                    'newton cuda': {'solver': 'newton', 'device': 'cuda'}},
    'RFECVSelector': {'liblinear': {},
                      'newton cpu': {'solver': 'newton'},
                      'newton cuda': {'solver': 'newton', 'device': 'cuda'}},
    'SequentialSelector': {'lbfgs': {},
                           'newton cpu': {'solver': 'newton'},
                           'newton cuda': {'solver': 'newton', 'device': 'cuda'}},
}


def load_data(synthetic, n_genes_max):
    """The training half of one split, Basal vs rest, scaled, float32."""
    if synthetic:
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(2976, n_genes_max)).astype(np.float32)
        y = rng.random(2976) < 0.2
        X[y, :50] += 0.5
    else:
        from featsel_datasets.scanb_gse96058.featsel_view import load_xy
        X_df, labels = load_xy()
        X, y = X_df.to_numpy(np.float32), (labels == 'Basal').to_numpy()
    split = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED)
    train, _ = next(split.split(X, y))
    X = StandardScaler().fit_transform(X[train]).astype(np.float32)
    return X, y[train].astype(int)


def variants_of(name, gpu_present):
    """The variants of a selector to run here: label -> extra parameters."""
    variants = VARIANTS.get(name, {'default': {}})
    return {label: params for label, params in variants.items()
            if gpu_present or params.get('device') != 'cuda'}


def time_fits(selector, X, y, repeats, max_seconds):
    """
    Seconds of each of `repeats` fits after one untimed fit, and the fitted
    selector. A warm-up fit longer than max_seconds is the only one: its
    time is returned instead, so a slow variant costs one fit, not several.
    """
    start = time.perf_counter()
    selector.fit(X, y)
    warm_up = time.perf_counter() - start
    if warm_up > max_seconds:
        return [warm_up], selector
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        selector.fit(X, y)
        seconds.append(time.perf_counter() - start)
    return seconds, selector


def measure(args, X_all, y, torch):
    """Yield one row per selector, device and size, as each is measured."""
    gpu_present = torch is not None and torch.cuda.is_available()
    genes = np.random.default_rng(SEED).permutation(X_all.shape[1])
    for name in args.selectors:
        cls = FeatureSelector.lookup(name)
        reference = {}   # the first variant's selection at each size
        for variant, extra in variants_of(name, gpu_present).items():
            for n_genes in sorted(args.n_genes):
                X = np.ascontiguousarray(X_all[:, np.sort(genes[:n_genes])])
                if gpu_present:
                    torch.cuda.reset_peak_memory_stats()
                seconds, fitted = time_fits(cls(**SELECTORS[name], **extra), X, y,
                                            args.repeats, args.max_seconds)
                support = fitted.get_support()
                reference.setdefault(n_genes, support)
                both = (support & reference[n_genes]).sum()
                either = (support | reference[n_genes]).sum()
                row = {
                    'selector': name, 'variant': variant,
                    'device': getattr(fitted, 'device_', 'cpu'), 'n_genes': n_genes,
                    'n_patients': X.shape[0], 'repeats': len(seconds),
                    'median_s': float(np.median(seconds)), 'min_s': min(seconds),
                    'max_s': max(seconds), 'n_selected': int(support.sum()),
                    'jaccard_vs_reference': both / either if either else np.nan,
                    'gpu_peak_mb': (torch.cuda.max_memory_allocated() / 2 ** 20
                                    if gpu_present else np.nan),
                }
                print(f"{name:24s} {variant:12s} {n_genes:6d} genes  "
                      f"median {row['median_s']:8.3f} s  "
                      f"[{row['min_s']:.3f}, {row['max_s']:.3f}]  "
                      f"jaccard {row['jaccard_vs_reference']:.3f}", flush=True)
                yield row
                if row['median_s'] > args.max_seconds:
                    print(f"{'':24s} {variant:12s} stops here: above {args.max_seconds} s")
                    break


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-genes', type=int, nargs='+', default=[1000, 3000, 10000, 28552])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--max-seconds', type=float, default=120.0,
                        help='Skip larger sizes of a selector once its median passes this')
    parser.add_argument('--selectors', nargs='+', default=list(SELECTORS))
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--max-memory-gb', type=float,
                        help='Cap on data memory; default 80%% of the RAM free at the start')
    parser.add_argument('--out', type=Path,
                        help='CSV to write; default results/local/benchmark_selectors/<start>.csv')
    args = parser.parse_args(argv)
    cap_gb = args.max_memory_gb or 0.8 * _available_gb()
    resource.setrlimit(resource.RLIMIT_DATA,
                       (int(cap_gb * 1024 ** 3), resource.getrlimit(resource.RLIMIT_DATA)[1]))
    args.max_memory_gb = round(cap_gb, 1)
    started = environment.now()
    out = args.out or OUT_DIR / f"{started[:19].replace(':', '')}.csv"   # 2026-11-07T001453.csv

    try:
        import torch
    except ImportError:
        torch = None

    X_all, y = load_data(args.synthetic, max(args.n_genes))
    settings = {'script': 'benchmark_selectors', 'started': started,
                'arguments': vars(args), 'selector_params': SELECTORS, 'variants': VARIANTS,
                'n_patients': X_all.shape[0], 'n_positive': int(y.sum())}
    environment.write_json(out.with_suffix('.json'), **settings, finished=None)
    print(f"{X_all.shape[0]} patients, {y.sum()} Basal; writing {out}")

    n_rows = 0
    with open(out, 'w', newline='') as handle:
        writer = None
        for row in measure(args, X_all, y, torch):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()   # an interrupted series keeps every row measured
            n_rows += 1

    environment.write_json(out.with_suffix('.json'), **settings, n_rows=n_rows,
                           finished=environment.now())
    print(f"Wrote {n_rows} rows to {out}")


def _available_gb():
    """MemAvailable from /proc/meminfo, in GB."""
    with open('/proc/meminfo') as handle:
        for line in handle:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError('No MemAvailable in /proc/meminfo')


if __name__ == '__main__':
    main()
