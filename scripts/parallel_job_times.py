"""
One pass of the scaling grid (scripts/parallel_scaling.py), keeping the runner's rows.

The rows hold each fit's own time. Comparing a 1-worker pass with a pass on
more workers shows whether jobs slow down when they run side by side, as
opposed to the speed-up being limited by the longest job or by serial work.

Usage:
    python scripts/parallel_job_times.py --n-jobs 1 --out final_report/figures/parallel_scaling_jobs_1_worker.csv
    python scripts/parallel_job_times.py --n-jobs 3 --out final_report/figures/parallel_scaling_jobs_3_workers.csv
"""

import argparse

from benchmark_speed import K_VALUES, LASSO, MODEL, N_SPLITS, RFE_PARAMS, SEED

from featsel.run import run
from featsel_datasets.scanb_gse96058.featsel_view import load_xy


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-jobs', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    X, y = load_xy(target='pam50 subtype')
    run(X, y, {
        'output': args.out, 'seed': SEED, 'n_splits': N_SPLITS, 'train_sizes': [0.5],
        'metrics': ['macro_f1'], 'selection_repeats': 1, 'n_jobs': args.n_jobs,
        'selectors': [{'name': 'LassoSelector', 'k': K_VALUES, 'params': LASSO},
                      {'name': 'RFESelector', 'k': K_VALUES, 'params': RFE_PARAMS}],
        'models': [{'name': 'logistic_regression', 'params': MODEL}],
    })


if __name__ == '__main__':
    main()
