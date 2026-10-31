"""
Compute the data facts the report states, into results/data_facts.csv.

docs/report/00_report_outline.md: every number in the report comes from a
results file, never retyped by hand. Each fact here has a name; a chapter
shows it with a marker that scripts/report_tables.py fills:

    <!-- fact: scanb_patients -->3,273<!-- /fact -->

Facts about the course's archived copies of SCAN-B need
references/local/scanb_course/, which is not in git. Without it, their
committed values are kept.

Usage:
    python scripts/data_facts.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, mannwhitneyu, normaltest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedShuffleSplit

from featsel_datasets.medmnist_breastmnist.breastmnist import BreastMNIST
from featsel_datasets.scanb_gse96058.featsel_view import load_xy
from featsel_datasets.scanb_gse96058.gse96058 import GSE96058
from featsel_datasets.scanb_gse96058.pam50 import PAM50_GENES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'data_facts.csv'
COURSE = ROOT / 'references' / 'local' / 'scanb_course'
SEED = 42


def count(n):
    return f'{int(n):,}'


def share(x, digits=0):
    return f'{100 * x:.{digits}f}%'


def number(x, digits=3):
    return f'{x:.{digits}f}'


def scanb_facts(data):
    """GSE96058 as published, and as this project loads it."""
    samples, expression = data.samples, data.expression
    hiseq = samples[samples['instrument model'] == 'HiSeq 2000']
    nextseq = samples[samples['instrument model'] == 'NextSeq 500']
    repeated = hiseq[hiseq['specimen'].duplicated(keep=False)]

    # Each repeated patient's two samples against each other, and the first
    # samples of different patients against each other: Pearson r over all
    # genes.
    first = hiseq[~hiseq['specimen'].duplicated()]
    same = [np.corrcoef(expression.loc[group.index].values)[0, 1]
            for _, group in repeated.groupby('specimen')]
    between = np.corrcoef(expression.loc[first.index].values)
    between = between[np.triu_indices_from(between, k=1)]
    pam50_same = repeated.groupby('specimen')['pam50 subtype'].nunique().eq(1).sum()

    er = hiseq[['er status', 'er prediction mgc']]
    known = er['er status'].notna()

    X, y = load_xy(data)
    floor = expression.loc[hiseq.index].iloc[:, :200]
    normal_p = normaltest(X.values, axis=0).pvalue
    normal = y.value_counts()['Normal']

    return {
        'scanb_patients': count(samples['specimen'].nunique()),
        'scanb_samples': count(len(samples)),
        'scanb_genes': count(expression.shape[1]),
        'scanb_hiseq_samples': count(len(hiseq)),
        'scanb_nextseq_samples': count(len(nextseq)),
        'scanb_nextseq_only_patients': count(len(set(nextseq['specimen']) - set(hiseq['specimen']))),
        'scanb_hiseq_patients': count(hiseq['specimen'].nunique()),
        'scanb_repeat_patients': count(repeated['specimen'].nunique()),
        'scanb_repeat_r_median': number(np.median(same)),
        'scanb_repeat_r_min': number(np.min(same)),
        'scanb_between_r_median': number(np.median(between)),
        'scanb_between_r_max': number(np.max(between)),
        'scanb_repeat_same_pam50': count(pam50_same),
        'scanb_er_known': count(known.sum()),
        'scanb_er_disagree': count((er.loc[known, 'er status'] != er.loc[known, 'er prediction mgc']).sum()),
        'scanb_er_missing': count((~known).sum()),
        'scanb_single_value_genes': count((expression.nunique() == 1).sum()),
        'scanb_floor_share_200': share((floor.values == floor.values.min()).mean()),
        'scanb_non_normal_share': share((normal_p < 0.05).mean(), 1),
        'scanb_median_excess_kurtosis': number(np.median(kurtosis(X.values, axis=0)), 2),
        'scanb_loaded_patients': count(len(y)),
        'scanb_loaded_genes': count(X.shape[1]),
        'scanb_normal_patients': count(normal),
        'scanb_rest_of_normal': count(len(y) - normal),
        'scanb_all_rest_accuracy': number(1 - normal / len(y), 2),
        'scanb_pam50_present': count(len(set(PAM50_GENES) & set(expression.columns))),
    }


def course_facts(data):
    """The course's archived copies, where this machine has them."""
    if not COURSE.exists():
        return {}
    meta = pd.read_csv(COURSE / 'scanb_full' / 'metadata.csv').set_index('samplename')
    # Samples are the columns of this file, so its genes are the row labels.
    small_genes = set(pd.read_csv(COURSE / 'scanb_small' / 'features.csv.gz',
                                  index_col=0, usecols=[0]).index)
    mgc = data.samples.loc[meta.index, 'er prediction mgc']
    return {
        'course_samples': count(len(meta)),
        'course_er_equals_mgc': count((meta['ER'] - 1 == mgc).sum()),
        'small_genes': count(len(small_genes)),
        'small_pam50_missing': count(len(set(PAM50_GENES) - small_genes)),
        'small_pam50_missing_list': ', '.join(sorted(set(PAM50_GENES) - small_genes)),
    }


def breastmnist_facts():
    data = BreastMNIST()
    X, samples = data.embeddings.values, data.samples
    labels = samples['label'].to_numpy()

    corr = np.corrcoef(X, rowvar=False)
    pairs = np.abs(corr[np.triu_indices_from(corr, k=1)])
    variance = np.linalg.svd(X - X.mean(axis=0), compute_uv=False) ** 2
    explained = np.cumsum(variance) / variance.sum()
    _, p = f_classif(X, labels)

    # Per training half of the comparison's splits: the share of features
    # passing the U test at a 5% false discovery rate, and at p < 0.05.
    bh, raw = [], []
    for train, _ in StratifiedShuffleSplit(5, train_size=0.5, random_state=SEED).split(X, labels):
        pos = X[train][labels[train] == 'malignant']
        neg = X[train][labels[train] != 'malignant']
        pu = np.sort(mannwhitneyu(pos, neg, axis=0).pvalue)
        m = len(pu)
        passed = np.flatnonzero(pu <= 0.05 * np.arange(1, m + 1) / m)
        bh.append((passed[-1] + 1 if len(passed) else 0) / m)
        raw.append(np.mean(pu < 0.05))

    sizes = samples['label'].value_counts()
    splits = samples['split'].value_counts()
    return {
        'bm_images': count(len(samples)),
        'bm_features': count(X.shape[1]),
        'bm_malignant': count(sizes['malignant']),
        'bm_benign_or_normal': count(sizes['benign_or_normal']),
        'bm_split_train': count(splits['train']),
        'bm_split_val': count(splits['val']),
        'bm_split_test': count(splits['test']),
        'bm_zero_share': share((X == 0).mean(), 1),
        'bm_median_abs_r': number(np.median(pairs)),
        'bm_pairs_above_half': share((pairs > 0.5).mean(), 2),
        'bm_pc_50': count(np.searchsorted(explained, 0.5) + 1),
        'bm_pc_90': count(np.searchsorted(explained, 0.9) + 1),
        'bm_pc_99': count(np.searchsorted(explained, 0.99) + 1),
        'bm_f_bonferroni': count((p < 0.05 / X.shape[1]).sum()),
        'bm_bh_share_min': share(min(bh)),
        'bm_bh_share_max': share(max(bh)),
        'bm_p05_share_min': share(min(raw)),
        'bm_p05_share_max': share(max(raw)),
    }


def comparison_facts():
    """Feature counts Higher Criticism chose in the selector comparisons."""
    facts = {}
    for dataset in ('scanb', 'breastmnist'):
        df = pd.read_csv(ROOT / 'results' / f'comparison_{dataset}.csv')
        for variant in ('HC2004', 'HC2008'):
            n = df.loc[df.selector == variant, 'n_selected']
            facts[f'{dataset}_{variant.lower()}_min'] = count(n.min())
            facts[f'{dataset}_{variant.lower()}_max'] = count(n.max())
    return facts


def main():
    data = GSE96058()
    facts = {**scanb_facts(data), **course_facts(data), **breastmnist_facts(),
             **comparison_facts()}
    if OUT.exists():  # keep committed facts this machine cannot recompute
        facts = {**pd.read_csv(OUT, index_col='name', keep_default_na=False).value.to_dict(),
                 **facts}
    table = pd.Series(facts, name='value').rename_axis('name')
    table.to_csv(OUT)
    print(table.to_string())


if __name__ == '__main__':
    main()
