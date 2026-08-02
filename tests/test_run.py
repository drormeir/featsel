"""
Tests for the config-driven experiment runner.

The runner is the entry point for every result in the report, so these tests
cover the contract that matters: one row per execution, no leakage of the
validation half into selection, and a resume that neither duplicates nor drops
work.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_classification

from featsel.run import KEY_COLUMNS, load_config, run


@pytest.fixture
def experiment(tmp_path):
    """A small dataset plus a config that exercises every axis."""
    X, y = make_classification(
        n_samples=120, n_features=200, n_informative=20, n_classes=3,
        n_clusters_per_class=1, random_state=0
    )
    ids = [f'sample{i}' for i in range(X.shape[0])]

    features = tmp_path / 'features.csv'
    metadata = tmp_path / 'metadata.csv'
    pd.DataFrame(X, index=ids,
                 columns=[f'gene{i}' for i in range(X.shape[1])]).T.to_csv(features)
    pd.DataFrame({'samplename': ids,
                  'label': [f'class{v}' for v in y]}).set_index('samplename').to_csv(metadata)

    dataset_config = tmp_path / 'data.yaml'
    dataset_config.write_text(yaml.safe_dump({
        'name': 'test', 'paths': {'features': str(features), 'metadata': str(metadata)},
        'sample_id_column': 'samplename', 'target_column': 'label',
        'transpose_features': True, 'separator': ',',
    }))

    experiment_config = tmp_path / 'experiment.yaml'
    experiment_config.write_text(yaml.safe_dump({
        'dataset': str(dataset_config),
        'output': str(tmp_path / 'out.csv'),
        'n_splits': 3,
        'train_sizes': [0.5],
        'preprocess': ['standard'],
        'metrics': ['accuracy', 'macro_f1', 'g_mean'],
        'selectors': [{'name': 'random', 'k': [5]}, {'name': 'anova_f', 'k': [5, 10]}],
        'models': [{'name': 'knn'}, {'name': 'lda', 'label': 'lda_shrinkage',
                                     'params': {'solver': 'lsqr', 'shrinkage': 'auto'}}],
    }))

    return load_config(str(experiment_config))


def test_one_row_per_execution(experiment):
    """Test that the grid produces exactly one row per axis combination."""
    out = run(experiment)
    df = pd.read_csv(out)

    expected = 3 * 1 * 1 * 3 * 2  # splits x train_sizes x preprocess x selector-k x models
    assert len(df) == expected
    assert not df.duplicated(KEY_COLUMNS).any()


def test_metrics_and_axes_recorded(experiment):
    """Test that every configured metric and axis lands in the CSV."""
    df = pd.read_csv(run(experiment))

    for column in ['accuracy', 'macro_f1', 'g_mean', *KEY_COLUMNS,
                   'seed', 'n_train', 'n_test', 'n_selected',
                   'selection_time_s', 'fit_time_s']:
        assert column in df.columns

    assert set(df.selector) == {'random', 'anova_f'}
    assert set(df.model) == {'knn', 'lda_shrinkage'}
    assert (df.n_selected == df.k).all()


def test_train_size_is_respected(experiment):
    """Test that train_size sets the split proportions."""
    experiment['train_sizes'] = [0.5]
    df = pd.read_csv(run(experiment))

    assert set(df.n_train) == {60}
    assert set(df.n_test) == {60}


def test_selection_differs_across_splits(experiment):
    """Test that selectors refit per split rather than reusing one selection."""
    df = pd.read_csv(run(experiment))

    for selector in ['random', 'anova_f']:
        picks = df[(df.selector == selector) & (df.k == 5)].selected_indices.unique()
        assert len(picks) > 1, f"{selector} chose identical features in every split"


def test_resume_completes_without_duplicating(experiment):
    """Test that resuming a truncated run fills the gap exactly once."""
    out = run(experiment)
    full = pd.read_csv(out)

    full.head(4).to_csv(out, index=False)
    run(experiment, resume=True)
    resumed = pd.read_csv(out)

    assert len(resumed) == len(full)
    assert not resumed.duplicated(KEY_COLUMNS).any()
    assert set(map(tuple, resumed[KEY_COLUMNS].astype(str).values)) == \
        set(map(tuple, full[KEY_COLUMNS].astype(str).values))


def test_resume_on_complete_run_is_a_noop(experiment):
    """Test that resuming a finished run adds nothing."""
    out = run(experiment)
    before = pd.read_csv(out)

    run(experiment, resume=True)
    after = pd.read_csv(out)

    assert len(after) == len(before)


def test_g_mean_is_zero_when_a_class_is_never_recalled():
    """Test that G-mean collapses to zero if any class is missed entirely."""
    from featsel.run import _g_mean

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 1])  # class 2 never recalled

    assert _g_mean(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)
    assert _g_mean(y_true, y_true) == pytest.approx(1.0)


def test_unknown_model_is_rejected(experiment):
    """Test that a typo in the config fails loudly."""
    experiment['models'] = [{'name': 'no_such_model'}]

    with pytest.raises(ValueError, match='Unknown entry'):
        run(experiment)
