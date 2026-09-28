"""
Tests for the config-driven experiment runner.

The runner is the entry point for every result in the report, so these tests
cover the contract that matters: one row per execution, no leakage of the
validation half into selection, and a resume that neither duplicates nor drops
work.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_classification

from featsel.analysis import kuncheva_index, summarize
from featsel.metrics import KEY_COLUMNS, g_mean
from featsel.run import build_tasks, load_config, run, run_config


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
        'selectors': [{'name': 'randomselector', 'k': [5]}, {'name': 'ANOVAFSelector', 'k': [5, 10]}],
        'models': [{'name': 'knn'}, {'name': 'lda', 'label': 'lda_shrinkage',
                                     'params': {'solver': 'lsqr', 'shrinkage': 'auto'}}],
    }))

    return load_config(str(experiment_config))


def test_one_row_per_execution(experiment):
    """Test that the grid produces exactly one row per axis combination."""
    out = run_config(experiment)
    df = pd.read_csv(out)

    expected = 1 * 3 * 1 * 1 * 3 * 2  # tasks x splits x train_sizes x preprocess x sel-k x models
    assert len(df) == expected
    assert not df.duplicated(KEY_COLUMNS).any()


class TestTaskFraming:
    """Tests for expanding one target into several classification tasks."""

    def test_multiclass_is_the_target_unchanged(self):
        """Test that the multiclass framing passes the target through."""
        y = np.array(['a', 'b', 'c', 'a'])
        tasks = build_tasks(y, ['multiclass'])

        assert len(tasks) == 1
        name, y_task, mask = tasks[0]
        assert name == 'multiclass'
        np.testing.assert_array_equal(y_task, y)
        assert mask.all()

    def test_one_vs_rest_gives_one_binary_task_per_class(self):
        """Test that one-vs-rest yields a binary task per class, using all samples."""
        y = np.array(['Basal', 'LumA', 'LumA', 'Normal'])
        tasks = build_tasks(y, ['one_vs_rest'])

        assert [name for name, _, _ in tasks] == ['ovr_Basal', 'ovr_LumA', 'ovr_Normal']
        for name, y_task, mask in tasks:
            assert set(np.unique(y_task)) == {name.removeprefix('ovr_'), 'rest'}
            assert mask.all()

        basal = next(y_task for name, y_task, _ in tasks if name == 'ovr_Basal')
        np.testing.assert_array_equal(basal, ['Basal', 'rest', 'rest', 'rest'])

    def test_one_vs_one_gives_a_task_per_pair_on_a_subset(self):
        """Test that one-vs-one yields every pair, restricted to those samples."""
        y = np.array(['Basal', 'LumA', 'LumA', 'Normal'])
        tasks = build_tasks(y, ['one_vs_one'])

        assert [name for name, _, _ in tasks] == [
            'ovo_Basal_vs_LumA', 'ovo_Basal_vs_Normal', 'ovo_LumA_vs_Normal'
        ]
        for name, y_task, mask in tasks:
            expected = set(name.removeprefix('ovo_').split('_vs_'))
            assert set(np.unique(y_task)) == expected
            assert mask.sum() == len(y_task)
            assert not mask.all()  # every pair excludes at least one sample here

        pair = next(t for t in tasks if t[0] == 'ovo_Basal_vs_Normal')
        np.testing.assert_array_equal(pair[1], ['Basal', 'Normal'])

    def test_one_vs_one_count_is_the_number_of_pairs(self):
        """Test that five classes give ten pairwise tasks."""
        y = np.array(['a', 'b', 'c', 'd', 'e'])
        assert len(build_tasks(y, ['one_vs_one'])) == 10

    def test_framings_combine(self):
        """Test that all three framings together give 1 + n + n(n-1)/2 tasks."""
        y = np.array(['a', 'b', 'c'])
        tasks = build_tasks(y, ['multiclass', 'one_vs_rest', 'one_vs_one'])

        assert len(tasks) == 1 + 3 + 3

    def test_unknown_framing_is_rejected(self):
        """Test that a typo in the framing list fails loudly."""
        with pytest.raises(ValueError, match='Unknown task framing'):
            build_tasks(np.array(['a', 'b']), ['leave_one_out'])

    def test_runner_records_each_task(self, experiment):
        """Test that every task appears in the CSV and rows do not collide."""
        experiment['task_framings'] = ['multiclass', 'one_vs_rest']
        df = pd.read_csv(run_config(experiment))

        assert set(df.task) == {'multiclass', 'ovr_class0', 'ovr_class1', 'ovr_class2'}
        assert not df.duplicated(KEY_COLUMNS).any()
        assert len(df) == 4 * 3 * 1 * 1 * 3 * 2

    def test_binary_tasks_score_differently_from_multiclass(self, experiment):
        """Test that the binary framings are genuinely different problems."""
        experiment['task_framings'] = ['multiclass', 'one_vs_rest']
        df = pd.read_csv(run_config(experiment))

        by_task = df.groupby('task').accuracy.mean()
        assert by_task['multiclass'] < by_task.drop('multiclass').max()

    def test_one_vs_one_uses_only_the_two_classes(self, experiment):
        """Test that pairwise tasks train and validate on a subset of samples."""
        experiment['task_framings'] = ['multiclass', 'one_vs_one']
        df = pd.read_csv(run_config(experiment))

        pairwise = df[df.task.str.startswith('ovo_')]
        multiclass = df[df.task == 'multiclass']

        assert set(pairwise.task) == {'ovo_class0_vs_class1', 'ovo_class0_vs_class2',
                                      'ovo_class1_vs_class2'}
        assert pairwise.n_train.max() < multiclass.n_train.iloc[0]
        assert not df.duplicated(KEY_COLUMNS).any()


def test_metrics_and_axes_recorded(experiment):
    """Test that every configured metric and axis lands in the CSV."""
    df = pd.read_csv(run_config(experiment))

    for column in ['accuracy', 'macro_f1', 'g_mean', *KEY_COLUMNS,
                   'seed', 'n_train', 'n_test', 'n_selected',
                   'selection_time_s', 'fit_time_s']:
        assert column in df.columns

    assert set(df.selector) == {'RandomSelector', 'ANOVAFSelector'}
    assert set(df.model) == {'knn', 'lda_shrinkage'}
    assert (df.n_selected == df.k).all()


def test_train_size_is_respected(experiment):
    """Test that train_size sets the split proportions."""
    experiment['train_sizes'] = [0.5]
    df = pd.read_csv(run_config(experiment))

    assert set(df.n_train) == {60}
    assert set(df.n_test) == {60}


def test_selection_differs_across_splits(experiment):
    """Test that selectors refit per split rather than reusing one selection."""
    df = pd.read_csv(run_config(experiment))

    for selector in ['RandomSelector', 'ANOVAFSelector']:
        picks = df[(df.selector == selector) & (df.k == 5)].selected_indices.unique()
        assert len(picks) > 1, f"{selector} chose identical features in every split"


def test_resume_completes_without_duplicating(experiment):
    """Test that resuming a truncated run fills the gap exactly once."""
    out = run_config(experiment)
    full = pd.read_csv(out)

    full.head(4).to_csv(out, index=False)
    run_config(experiment, resume=True)
    resumed = pd.read_csv(out)

    assert len(resumed) == len(full)
    assert not resumed.duplicated(KEY_COLUMNS).any()
    assert set(map(tuple, resumed[KEY_COLUMNS].astype(str).values)) == \
        set(map(tuple, full[KEY_COLUMNS].astype(str).values))


def test_resume_on_complete_run_is_a_noop(experiment):
    """Test that resuming a finished run adds nothing."""
    out = run_config(experiment)
    before = pd.read_csv(out)

    run_config(experiment, resume=True)
    after = pd.read_csv(out)

    assert len(after) == len(before)


def test_g_mean_is_zero_when_a_class_is_never_recalled():
    """Test that G-mean collapses to zero if any class is missed entirely."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 1])  # class 2 never recalled

    assert g_mean(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)
    assert g_mean(y_true, y_true) == pytest.approx(1.0)


def test_run_takes_arrays_in_memory(experiment):
    """Test that run() needs no dataset file and applies config defaults."""
    X, y = make_classification(n_samples=60, n_features=50, n_informative=5,
                               n_classes=2, random_state=0)
    config = {key: experiment[key] for key in ('output', 'selectors', 'models')}
    config['n_splits'] = 2

    df = pd.read_csv(run(X, y, config))

    assert len(df) == 1 * 2 * 1 * 1 * 3 * 2  # default train size, preprocess, framing
    assert set(df.train_size) == {0.5}


def test_parallel_run_matches_sequential(experiment, tmp_path):
    """Test that the number of workers changes nothing but the runtime."""
    sequential = pd.read_csv(run_config(experiment))

    experiment.update(n_jobs=2, output=str(tmp_path / 'parallel.csv'))
    parallel = pd.read_csv(run_config(experiment))

    timing = [c for c in sequential.columns if c.endswith(('_time_s', '_peak_mb'))]
    pd.testing.assert_frame_equal(sequential.drop(columns=timing),
                                  parallel.drop(columns=timing))


def test_truth_records_feature_recovery(experiment):
    """Test that known informative features yield recall and precision per row."""
    X, y = make_classification(n_samples=120, n_features=200, n_informative=10,
                               n_redundant=0, shuffle=False, random_state=0)
    truth = np.arange(10)  # shuffle=False puts the informative columns first
    config = {key: experiment[key] for key in ('output', 'selectors', 'models')}
    config['n_splits'] = 3

    df = pd.read_csv(run(X, y, config, truth=truth))

    assert df.truth_recall.between(0, 1).all()
    assert df.truth_precision.between(0, 1).all()
    by_selector = df.groupby('selector').truth_precision.mean()
    assert by_selector['ANOVAFSelector'] > by_selector['RandomSelector']


def test_config_paths_are_relative_to_the_config_file(tmp_path, monkeypatch):
    """Test that dataset and output resolve against the config, not the cwd."""
    (tmp_path / 'configs').mkdir()
    path = tmp_path / 'configs' / 'exp.yaml'
    path.write_text(yaml.safe_dump({'dataset': 'data.yaml', 'output': '../results/out.csv'}))
    monkeypatch.chdir('/')

    config = load_config(str(path))

    assert Path(config['dataset']).resolve() == tmp_path / 'configs' / 'data.yaml'
    assert Path(config['output']).resolve() == tmp_path / 'results' / 'out.csv'


def test_summarize_gives_one_row_per_cell(experiment):
    """Test that summarize collapses the splits of every cell into one row."""
    df = pd.read_csv(run_config(experiment))
    summary = summarize(df)

    assert len(summary) == 3 * 2  # sel-k x models
    for column in ['macro_f1_median', 'macro_f1_q1', 'macro_f1_q3', 'selection_time_s_median']:
        assert column in summary.columns
    assert (summary.macro_f1_q1 <= summary.macro_f1_q3).all()


def test_kuncheva_separates_random_from_anova(experiment):
    """Test that random selection scores near zero and ANOVA F well above it."""
    df = pd.read_csv(run_config(experiment))
    stability = kuncheva_index(df, n_total_features=200).set_index(['selector', 'k'])

    assert len(stability) == 3  # sel-k, the model does not change the selection
    assert (stability.n_pairs == 3).all()  # 3 splits give 3 pairs
    assert stability.loc[('ANOVAFSelector', 5), 'consistency_index'] > \
        stability.loc[('RandomSelector', 5), 'consistency_index'] + 0.3


def test_unknown_model_is_rejected(experiment):
    """Test that a typo in the config fails loudly."""
    experiment['models'] = [{'name': 'no_such_model'}]

    with pytest.raises(ValueError, match='Unknown entry'):
        run_config(experiment)
