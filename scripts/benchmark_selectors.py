"""
Time every selector against its original, on SCAN-B, on one machine.

The series behind any speed claim about a selector. featsel.benchmark does
the work and defines each selector's variants: its original first
(scikit-learn's version where one exists), then this package's solvers,
threads and devices. Per variant and gene count a row gives the median and
range of the timed fits, whether every fit chose the same features, and the
Jaccard index of its features against the original's.

The process's data memory is capped (--max-memory-gb, by default 80% of
the RAM free at the start), so a fit that would exhaust the machine raises
MemoryError and ends the series instead.

The data is SCAN-B, Basal vs rest, the training half of one split at train
fraction 0.5, scaled on that half and held in float32, as the runner holds
it. Genes are a random subset, the same for every selector; each larger
subset contains the smaller ones. --synthetic replaces SCAN-B by
featsel.benchmark's synthetic data, to check the script end to end.

Each selector's rows are appended to a CSV as soon as it finishes, so an
interrupted series keeps what it finished. Beside it a JSON file records the
arguments, the machine, the library versions and the git commit. Both go to
results/local/benchmark_selectors/, which git ignores, named by start time.

Usage:
    python scripts/benchmark_selectors.py --synthetic --n-genes 500 2000 --repeats 2
    python scripts/benchmark_selectors.py --selectors RFESelector --n-genes 1000 3000
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from featsel import benchmark, environment

SEED = benchmark.SEED
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'results' / 'local' / 'benchmark_selectors'


def load_data(synthetic, n_genes_max):
    """The training half of one split, Basal vs rest, scaled, float32."""
    if synthetic:
        return benchmark.synthetic(n_genes=n_genes_max)
    from featsel_datasets.scanb_gse96058.featsel_view import load_xy
    X_df, labels = load_xy()
    X, y = X_df.to_numpy(np.float32), (labels == 'Basal').to_numpy()
    split = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED)
    train, _ = next(split.split(X, y))
    X = StandardScaler().fit_transform(X[train]).astype(np.float32)
    return X, y[train].astype(int)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-genes', type=int, nargs='+', default=[1000, 3000, 10000, 28541])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--max-seconds', type=float, default=120.0,
                        help='Skip larger sizes of a variant once its first fit passes this')
    parser.add_argument('--selectors', nargs='+', default=list(benchmark.PARAMS))
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--max-memory-gb', type=float,
                        help='Cap on data memory; default 80%% of the RAM free at the start')
    parser.add_argument('--out', type=Path,
                        help='CSV to write; default results/local/benchmark_selectors/<start>.csv')
    args = parser.parse_args(argv)
    if args.max_memory_gb:
        benchmark.cap_memory(gb=args.max_memory_gb)
    else:
        args.max_memory_gb = round(benchmark.cap_memory(), 1)
    started = environment.now()
    out = args.out or OUT_DIR / f"{started[:19].replace(':', '')}.csv"   # 2026-11-07T001453.csv
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.synthetic, max(args.n_genes))
    X = X[:, np.random.default_rng(SEED).permutation(X.shape[1])]   # random gene subsets
    settings = {'script': 'benchmark_selectors', 'started': started,
                'arguments': vars(args), 'selector_params': benchmark.PARAMS,
                'variants': benchmark.VARIANTS,
                'n_patients': X.shape[0], 'n_positive': int(y.sum())}
    environment.write_json(out.with_suffix('.json'), **settings, finished=None)
    print(f"{X.shape[0]} patients, {y.sum()} positive; writing {out}")

    n_rows = 0
    for name in args.selectors:
        table = benchmark.benchmark(name, X, y, n_genes=args.n_genes, repeats=args.repeats,
                                    max_seconds=args.max_seconds)
        table.insert(0, 'selector', name)
        table.to_csv(out, mode='a', header=n_rows == 0, index=False)
        n_rows += len(table)
        print(table[['selector', 'variant', 'device', 'n_genes', 'median_s', 'reproducible',
                     'jaccard_vs_original']].to_string(index=False, header=False), flush=True)

    environment.write_json(out.with_suffix('.json'), **settings, n_rows=n_rows,
                           finished=environment.now())
    print(f"Wrote {n_rows} rows to {out}")


if __name__ == '__main__':
    main()
