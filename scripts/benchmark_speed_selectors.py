"""
Time selectors against their originals as a plain script (featsel.benchmark).

The speed notebooks run the same benchmark inside Jupyter, where Newton
RFECV on the CPU took about 1.5 times longer than in a plain script, for
no reason found. So timings for the report come from this script.

For each selector named, the synthetic SCAN-B-shaped series goes to
results/local/speed/<name>.csv. With --scanb, the five PAM50 tasks of real
SCAN-B go to <name>_scanb.csv. Each CSV has a JSON beside it with the
machine.

Usage:
    python scripts/benchmark_speed_selectors.py RFESelector RFECVSelector
    python scripts/benchmark_speed_selectors.py LassoSelector --scanb
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from featsel import benchmark, environment
from featsel.run import build_tasks

OUT = Path(__file__).resolve().parents[1] / 'results' / 'local' / 'speed'


def save(table, path, name, **details):
    """Write the table and a JSON of the machine beside it."""
    table.to_csv(path, index=False)
    environment.write_json(path.with_suffix('.json'), selector=name,
                           params=benchmark.PARAMS[name], **details)
    print(table.drop(columns=['min_s', 'max_s'], errors='ignore').round(3).to_string(index=False),
          flush=True)


def scanb_series(name):
    """The five PAM50 tasks, all patients and genes, standardized in 32 bits as the runner does."""
    from featsel_datasets.scanb_gse96058.featsel_view import load_xy

    X_df, y = load_xy()
    X = StandardScaler().fit_transform(X_df.to_numpy(dtype=np.float32)).astype(np.float32)
    rows = []
    for task, y_task in build_tasks(y.to_numpy()):
        positive = y_task != 'rest'
        part = benchmark.benchmark(name, X, positive.astype(int), n_genes=(X.shape[1],))
        part.insert(0, 'task', task)
        part.insert(1, 'n_positive', int(positive.sum()))
        rows.append(part)
    return pd.concat(rows, ignore_index=True), X.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='+', help='Selector class names')
    parser.add_argument('--scanb', action='store_true', help='Also time the five SCAN-B tasks')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Memory cap: {benchmark.cap_memory():.1f} GB", flush=True)
    X, y = benchmark.synthetic()
    for name in args.names:
        print(f"\n=== {name}, synthetic", flush=True)
        save(benchmark.benchmark(name, X, y), OUT / f'{name}.csv', name,
             n_patients=int(X.shape[0]))
        if args.scanb:
            print(f"\n=== {name}, SCAN-B", flush=True)
            table, n_patients = scanb_series(name)
            save(table, OUT / f'{name}_scanb.csv', name, n_patients=n_patients,
                 dataset='SCAN-B GSE96058, PAM50 one-vs-rest, standardized')


if __name__ == '__main__':
    main()
