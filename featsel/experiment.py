"""
Evaluation harness for comparing feature selection methods.

One protocol for every method: stratified k-fold cross-validation, with the
selector fit on the training split only. For each (selector, k, classifier)
cell it records predictive metrics, selection runtime, peak selection memory,
and the selected feature indices, so selection stability can be computed
afterwards.

Adding a classifier means adding one entry to CLASSIFIERS. Adding a selector
means adding one entry to SELECTORS.
"""

import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .feature_selector import FeatureSelector

# name -> callable(seed) -> unfitted sklearn estimator.
# Adding a classifier to the study is one line here.
CLASSIFIERS = {
    'logistic_regression': lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    'linear_svm': lambda seed: LinearSVC(max_iter=20000, random_state=seed),
    'random_forest': lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
    'knn': lambda seed: KNeighborsClassifier(n_neighbors=5),
    'lda_shrinkage': lambda seed: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
}

try:  # optional dependency, reported as an implementation comparison
    from xgboost import XGBClassifier

    CLASSIFIERS['xgboost'] = lambda seed: XGBClassifier(
        n_estimators=200, tree_method='hist', random_state=seed,
        n_jobs=1, verbosity=0,
    )
except ImportError:  # pragma: no cover - depends on the environment
    pass

# name -> extra kwargs passed to FeatureSelector alongside method and n_features.
SELECTORS = {
    'random': {'method': 'random'},
    'anova_f': {'method': 'anova_f'},
}

K_VALUES = (10, 25, 50, 100, 250, 500, 1000)


def _fold_metrics(y_true, y_pred):
    """Predictive metrics for one fold, all suited to imbalanced classes."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }


def _make_preprocessor(name, seed):
    """
    Build the per-split preprocessing step.

    'standard'        : centre and scale, leaving the shape of each column alone.
    'quantile_normal' : map each column onto a normal distribution by rank,
                        which is what ANOVA F and LDA assume and what trees are
                        invariant to.
    """
    if name == 'standard':
        return StandardScaler()
    if name == 'quantile_normal':
        return QuantileTransformer(
            output_distribution='normal', subsample=100_000, random_state=seed
        )
    raise ValueError(f"Unknown preprocess '{name}'. "
                     "Available: 'standard', 'quantile_normal'")


def run_grid(X, y, selectors=None, k_values=None, classifiers=None,
             n_splits=5, seed=42, preprocess='standard', verbose=True):
    """
    Run the full (selector x k x classifier x fold) grid.

    Every selector is fit inside the fold, on the training split only.
    Imputation and scaling are fit on the training split too.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : pd.Series or np.ndarray of shape (n_samples,)
        Target labels.
    selectors : dict, optional
        Mapping of selector name to FeatureSelector kwargs. Defaults to SELECTORS.
    k_values : iterable of int, optional
        Feature counts to sweep. Defaults to K_VALUES.
    classifiers : dict, optional
        Mapping of classifier name to a callable taking a seed. Defaults to CLASSIFIERS.
    n_splits : int, default=5
        Number of stratified cross-validation folds.
    seed : int, default=42
        Random seed, used for the splitter, the selectors and the classifiers.
    preprocess : str, default='standard'
        Per-split preprocessing: 'standard' or 'quantile_normal'.
    verbose : bool, default=True
        Print progress per fold.

    Returns
    -------
    scores : pd.DataFrame
        One row per (selector, k, classifier, fold), with predictive metrics,
        selection_time_s, selection_peak_mb and n_selected.
    supports : pd.DataFrame
        One row per (selector, k, fold), with the selected feature indices as a
        tuple. Input to selection stability measures.
    """
    selectors = SELECTORS if selectors is None else selectors
    k_values = K_VALUES if k_values is None else k_values
    classifiers = CLASSIFIERS if classifiers is None else classifiers

    X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    score_rows, support_rows = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
        if verbose:
            print(f"fold {fold + 1}/{n_splits}", flush=True)

        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        # Preprocessing is part of the fold: fit on train, apply to test.
        imputer = SimpleImputer(strategy='median').fit(X_train)
        scaler = _make_preprocessor(preprocess, seed).fit(imputer.transform(X_train))
        X_train = scaler.transform(imputer.transform(X_train))
        X_test = scaler.transform(imputer.transform(X_test))

        for sel_name, sel_kwargs in selectors.items():
            for k in k_values:
                if k > X_train.shape[1]:
                    continue

                # Per-fold seed: a stochastic selector must redraw in every
                # fold, or its selection stability is trivially 1.0.
                selector = FeatureSelector(
                    n_features=k, random_state=seed + fold, **sel_kwargs
                )

                tracemalloc.start()
                start = time.perf_counter()
                selector.fit(X_train, y_train)
                selection_time = time.perf_counter() - start
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                support = selector.get_support(indices=True)
                X_train_sel = X_train[:, support]
                X_test_sel = X_test[:, support]

                support_rows.append({
                    'selector': sel_name,
                    'k': k,
                    'fold': fold,
                    'n_selected': len(support),
                    'selected_indices': tuple(int(i) for i in support),
                })

                for clf_name, make_clf in classifiers.items():
                    clf = make_clf(seed)
                    start = time.perf_counter()
                    clf.fit(X_train_sel, y_train)
                    fit_time = time.perf_counter() - start
                    y_pred = clf.predict(X_test_sel)

                    score_rows.append({
                        'selector': sel_name,
                        'k': k,
                        'classifier': clf_name,
                        'fold': fold,
                        'n_selected': len(support),
                        'selection_time_s': selection_time,
                        'selection_peak_mb': peak_bytes / 1024 ** 2,
                        'classifier_fit_time_s': fit_time,
                        **_fold_metrics(y_test, y_pred),
                    })

    return pd.DataFrame(score_rows), pd.DataFrame(support_rows)


def kuncheva_index(supports, n_total_features):
    """
    Selection stability across folds, corrected for chance agreement.

    Kuncheva's consistency index for a pair of equally sized subsets:

        I = (r - k^2 / n) / (k - k^2 / n)

    where r is the size of the intersection, k the subset size and n the total
    number of features. It is 1 for identical subsets and 0 for the overlap
    expected by chance, which is why a random selector scores near 0 here even
    though its raw Jaccard overlap is positive. Reported as the mean over all
    fold pairs.

    Parameters
    ----------
    supports : pd.DataFrame
        Output of run_grid(), with columns selector, k, fold, selected_indices.
    n_total_features : int
        Total number of features available to the selector.

    Returns
    -------
    stability : pd.DataFrame
        One row per (selector, k) with the mean pairwise consistency index.
    """
    rows = []

    for (sel_name, k), group in supports.groupby(['selector', 'k']):
        subsets = [set(s) for s in group['selected_indices']]
        sizes = {len(s) for s in subsets}

        if len(subsets) < 2:
            continue

        pair_scores = []
        for i in range(len(subsets)):
            for j in range(i + 1, len(subsets)):
                # Use the actual subset sizes; they match unless a selector
                # picks its own feature count (e.g. Higher Criticism).
                k_eff = (len(subsets[i]) + len(subsets[j])) / 2
                expected = k_eff ** 2 / n_total_features
                denominator = k_eff - expected
                if denominator <= 0:
                    continue
                r = len(subsets[i] & subsets[j])
                pair_scores.append((r - expected) / denominator)

        if pair_scores:
            rows.append({
                'selector': sel_name,
                'k': k,
                'consistency_index': float(np.mean(pair_scores)),
                'n_pairs': len(pair_scores),
                'variable_size': len(sizes) > 1,
            })

    return pd.DataFrame(rows)


def summarize(scores):
    """
    Aggregate fold-level scores into one row per (selector, k, classifier).

    Parameters
    ----------
    scores : pd.DataFrame
        Fold-level output of run_grid().

    Returns
    -------
    summary : pd.DataFrame
        Mean and standard deviation over folds for every metric.
    """
    metrics = ['accuracy', 'macro_f1', 'balanced_accuracy', 'mcc',
               'selection_time_s', 'selection_peak_mb', 'classifier_fit_time_s',
               'n_selected']

    grouped = scores.groupby(['selector', 'k', 'classifier'])[metrics]
    summary = grouped.agg(['mean', 'std'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    return summary.reset_index()
