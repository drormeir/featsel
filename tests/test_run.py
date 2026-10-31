"""
Tests for the config-driven experiment runner.

The runner is the entry point for every result in the report, so these tests
cover the contract that matters: one row per execution, no leakage of the
validation half into selection, and a resume that neither duplicates nor drops
work.
"""

import json
import resource
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from joblib.externals.loky.process_executor import TerminatedWorkerError
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import featsel.run as run_module
from featsel.analysis import kuncheva_index, summarize
from featsel.metrics import KEY_COLUMNS, g_mean
from featsel.run import build_tasks, load_config, load_dataset, run, run_config
from featsel.selectors import ANOVAFSelector, FeatureSelector


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

    expected = 3 * 3 * 1 * 1 * 3 * 2  # tasks x splits x train_sizes x preprocess x sel-k x models
    assert len(df) == expected
    assert not df.duplicated(KEY_COLUMNS).any()


class TestTasks:
    """Tests for expanding one target into binary classification tasks."""

    def test_binary_target_is_one_task_unchanged(self):
        """Test that a two-class target passes through as a single task."""
        y = np.array(['a', 'b', 'b', 'a'])
        tasks = build_tasks(y)

        assert len(tasks) == 1
        name, y_task = tasks[0]
        assert name == 'binary'
        np.testing.assert_array_equal(y_task, y)

    def test_multiclass_target_gives_one_task_per_class(self):
        """Test that each class becomes a task against the rest."""
        y = np.array(['Basal', 'LumA', 'LumA', 'Normal'])
        tasks = build_tasks(y)

        assert [name for name, _ in tasks] == ['ovr_Basal', 'ovr_LumA', 'ovr_Normal']
        for name, y_task in tasks:
            assert set(np.unique(y_task)) == {name.removeprefix('ovr_'), 'rest'}

        basal = next(y_task for name, y_task in tasks if name == 'ovr_Basal')
        np.testing.assert_array_equal(basal, ['Basal', 'rest', 'rest', 'rest'])

    def test_runner_records_each_task(self, experiment):
        """Test that every task appears in the CSV and rows do not collide."""
        df = pd.read_csv(run_config(experiment))

        assert set(df.task) == {'ovr_class0', 'ovr_class1', 'ovr_class2'}
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

    assert len(df) == 1 * 2 * 1 * 1 * 3 * 2  # one binary task, default train size, preprocess
    assert set(df.train_size) == {0.5}


GRID = {'C': [0.01, 1.0, 100.0]}


class TestTuning:
    """Tests for choosing a model's parameters on inner splits of the training half."""

    @pytest.fixture
    def tuned(self, experiment):
        X, y = make_classification(n_samples=120, n_features=50, n_informative=5,
                                   n_classes=2, random_state=0)
        config = {
            'output': experiment['output'],
            'n_splits': 2,
            'selectors': [{'name': 'ANOVAFSelector', 'k': [10]}],
            'models': [{'name': 'logistic_regression', 'grid': GRID},
                       {'name': 'knn'}],
        }
        return X, y, pd.read_csv(run(X, y, config))

    def test_tuned_values_are_recorded(self, tuned):
        """Test that a tuned model records its choice and an untuned one records none."""
        _, _, df = tuned
        chosen = df.loc[df.model == 'logistic_regression', 'tuned_params'].map(json.loads)

        assert all(c['C'] in GRID['C'] for c in chosen)
        assert set(df.loc[df.model == 'knn', 'tuned_params']) == {'{}'}

    def test_choice_uses_the_training_half_only(self, tuned):
        """Test that the choice and score match tuning on the training half by hand."""
        X, y, df = tuned
        train, test = next(StratifiedShuffleSplit(n_splits=2, train_size=0.5,
                                                  random_state=42).split(X, y))
        scaler = StandardScaler().fit(X[train])
        selector = ANOVAFSelector(n_features=10).fit(scaler.transform(X[train]), y[train])
        X_train = selector.transform(scaler.transform(X[train]))
        X_test = selector.transform(scaler.transform(X[test]))
        inner = StratifiedShuffleSplit(n_splits=3, train_size=0.5, random_state=42)
        search = GridSearchCV(LogisticRegression(random_state=42), GRID,
                              scoring='roc_auc', cv=inner).fit(X_train, y[train])

        row = df[(df.model == 'logistic_regression') & (df.split == 0)].iloc[0]
        assert json.loads(row.tuned_params) == search.best_params_
        assert row.accuracy == accuracy_score(y[test], search.predict(X_test))


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


def test_ranking_once_selects_the_same_genes_as_one_fit_per_k(experiment, monkeypatch):
    """Test that cutting one ranking at each k equals fitting once per k."""
    experiment['selectors'] = [
        {'name': 'LassoSelector', 'k': [5, 10, 20], 'params': {'C': 1.0}},
        {'name': 'TreeImportanceSelector', 'k': [5, 10, 20], 'params': {'n_estimators': 20}},
        {'name': 'RFESelector', 'k': [5, 10, 20, 100], 'params': {'step': 0.3}},
    ]
    once = pd.read_csv(run_config(experiment))
    monkeypatch.setattr(run_module, 'RANK_ONCE', set())
    per_k = pd.read_csv(run_config(experiment))

    pd.testing.assert_series_equal(once['selected_indices'], per_k['selected_indices'])
    assert (once['n_selected'] == once['k']).all()


def test_dataset_can_be_a_module(tmp_path, monkeypatch):
    """Test that a dataset module supplies X, y, its default target and its name."""
    (tmp_path / 'toy_view.py').write_text(
        "import pandas as pd\n"
        "NAME = 'toy'\n"
        "TARGET = 'label'\n"
        "def load_xy(target=TARGET):\n"
        "    X = pd.DataFrame({'g0': [1.0, 2.0], 'g1': [0.0, 1.0]})\n"
        "    return X, pd.Series(['a', 'b'], name=target)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / 'exp.yaml'
    path.write_text(yaml.safe_dump({'dataset': 'toy_view', 'output': 'out.csv'}))

    config = load_config(str(path))
    assert config['dataset'] == 'toy_view'  # a module name is not a path

    X, y, target, name = load_dataset(config)
    assert (target, name) == ('label', 'toy')
    assert X['g0'].tolist() == [1.0, 2.0]
    _, y, target, _ = load_dataset({**config, 'target': 'other'})
    assert target == 'other' and y.name == 'other'


def test_summarize_gives_one_row_per_cell(experiment):
    """Test that summarize collapses the splits of every cell into one row."""
    df = pd.read_csv(run_config(experiment))
    summary = summarize(df)

    assert len(summary) == 3 * 3 * 2  # tasks x sel-k x models
    for column in ['macro_f1_median', 'macro_f1_q1', 'macro_f1_q3', 'selection_time_s_median']:
        assert column in summary.columns
    assert (summary.macro_f1_q1 <= summary.macro_f1_q3).all()


def test_kuncheva_separates_random_from_anova(experiment):
    """Test that random selection scores near zero and ANOVA F well above it."""
    df = pd.read_csv(run_config(experiment))
    stability = (kuncheva_index(df, n_total_features=200)
                 .set_index(['selector', 'k', 'task']).sort_index())

    assert len(stability) == 3 * 3  # sel-k x tasks, the model does not change the selection
    assert (stability.n_pairs == 3).all()  # 3 splits give 3 pairs
    anova = stability.loc[('ANOVAFSelector', 5), 'consistency_index']
    random = stability.loc[('RandomSelector', 5), 'consistency_index']
    assert (anova > random + 0.2).all()


def test_unknown_model_is_rejected(experiment):
    """Test that a typo in the config fails loudly."""
    experiment['models'] = [{'name': 'no_such_model'}]

    with pytest.raises(ValueError, match='Unknown entry'):
        run_config(experiment)


class GreedySelector(FeatureSelector):
    """A selector that asks for 2 GB, to hit the memory cap for real."""

    def __init__(self, check_input=True):
        self.check_input = check_input

    def fit(self, X, y):
        X, y = self._validate(X, y)
        np.ones(2 ** 28)  # 2 GB of float64
        self.support_ = self._mask([0])
        return self


def _address_space_gb():
    """This process's current virtual memory, from /proc."""
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmSize:'):
            return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError('No VmSize in /proc/self/status')


class TestMemoryCap:
    """Tests for max_worker_memory_gb: a cell past the cap is skipped, not killed."""

    def test_cell_past_the_cap_is_skipped_and_reported(self, experiment, capsys):
        """Test that the greedy selector's cells are skipped and the rest written."""
        experiment['selectors'] = [{'name': 'GreedySelector'}, {'name': 'ANOVAFSelector', 'k': [5]}]
        experiment['max_worker_memory_gb'] = _address_space_gb() + 0.5

        df = pd.read_csv(run_config(experiment))

        assert set(df.selector) == {'ANOVAFSelector'}
        assert len(df) == 3 * 3 * 1 * 2  # tasks x splits x sel-k x models
        out = capsys.readouterr().out
        assert 'Out of memory, skipped (9)' in out
        assert 'GreedySelector k=None' in out

    def test_cap_is_lifted_after_the_split(self, experiment):
        """Test that the process gets its old limit back, so later work is not capped."""
        before = resource.getrlimit(resource.RLIMIT_AS)
        experiment['max_worker_memory_gb'] = _address_space_gb() + 2

        run_config(experiment)

        assert resource.getrlimit(resource.RLIMIT_AS) == before


def test_dead_worker_stops_with_a_message(experiment, monkeypatch):
    """Test that a worker killed mid-run ends the CLI with a short message, rows kept."""
    real_iter_rows = run_module.iter_rows

    def dies_after_one_row(*args, **kwargs):
        yield next(real_iter_rows(*args, **kwargs))
        raise TerminatedWorkerError('worker aborted')

    monkeypatch.setattr(run_module, 'iter_rows', dies_after_one_row)
    monkeypatch.setattr(run_module, 'load_config', lambda path: experiment)

    with pytest.raises(SystemExit) as stopped:
        run_module.main(['--config', 'unused.yaml'])

    assert 'most likely out of memory' in str(stopped.value.code)
    assert '1 executions are saved' in str(stopped.value.code)
    assert len(pd.read_csv(experiment['output'])) == 1


class TestSharedSplits:
    """Tests for preparing each split once and sharing it across tasks and workers."""

    def test_every_task_uses_the_same_split(self, experiment, monkeypatch):
        """Test that all tasks of a split read one prepared copy of the same patients."""
        real_job_rows = run_module._job_rows
        seen = {}

        def recording_job_rows(folder, y_train, y_test, context, *args):
            seen.setdefault((context['split'], context['task']), set()).add(folder)
            return real_job_rows(folder, y_train, y_test, context, *args)

        monkeypatch.setattr(run_module, '_job_rows', recording_job_rows)
        run_config(experiment)

        for split in range(3):
            folders = {folder for (s, _), group in seen.items() if s == split
                       for folder in group}
            assert len(folders) == 1
        assert {task for _, task in seen} == {'ovr_class0', 'ovr_class1', 'ovr_class2'}

    def test_shared_files_are_removed(self, experiment):
        """Test that no prepared split is left in shared memory after a run."""
        shm = Path('/dev/shm')
        before = set(shm.glob('featsel_*'))

        run_config(experiment)

        assert set(shm.glob('featsel_*')) == before

    def test_rank_once_selectors_keep_their_k_values_together(self):
        """Test that Lasso fits once for all k, while ANOVA gets one job per k."""
        lasso = run_module._k_groups({'name': 'LassoSelector', 'k': [5, 10]})
        anova = run_module._k_groups({'name': 'ANOVAFSelector', 'k': [5, 10]})
        hc = run_module._k_groups({'name': 'HigherCriticismSelector'})

        assert lasso == [[5, 10]]
        assert anova == [[5], [10]]
        assert hc == [[None]]


def test_resume_with_a_selector_without_k_does_not_duplicate(experiment):
    """Test that an empty k in the CSV, and the float column it causes, still match on resume."""
    experiment['selectors'] = [{'name': 'HigherCriticismSelector'},
                               {'name': 'ANOVAFSelector', 'k': [5, 10]}]
    first = pd.read_csv(run_config(experiment))

    again = pd.read_csv(run_config(experiment, resume=True))

    assert len(again) == len(first)
    assert not again.duplicated(KEY_COLUMNS).any()
