# Architecture options

Four candidate designs for how `featsel` is used. Each is shown on the same
uses: in-memory data, files, the command line. These sketch the desired tool,
not the current code.

Written 2026-09-26.

## Background

`featsel` is a general tool for feature selection on high-dimensional data.
High dimensional means many features relative to samples. With that many
features, a model can fit noise. Keeping a small, informative subset often
predicts as well or better. It is also cheaper and easier to interpret.

The tool does two things for any dataset. It selects features with a chosen
method. It also compares selection methods under one protocol, against a
random-selection baseline.

The M.Sc. final project is one use of the tool. It applies the comparison to
SCAN-B (about 9,000 genes, about 3,000 patients) and to a second dataset. Its
research question: for a given model and a given number of features, which selection
method works best, and by how much over picking features at random?

## Terms

- **Sample.** One row of the data. In SCAN-B, one patient.
- **Feature.** One column. In SCAN-B, one gene's expression level.
- **`X`.** The feature matrix, samples by features.
- **`y`.** The label of each sample. In SCAN-B, the PAM50 subtype.
- **Ground truth.** The indices of the features that truly carry signal. Known
  only for simulated data.
- **Selector.** An object that picks which features to keep. It is fitted on
  `X, y`, scores or ranks every feature, and keeps a subset. `transform(X)`
  then returns only those columns. It follows the scikit-learn transformer
  interface (`fit`, `transform`, `get_support`).
- **`k`.** The feature budget: how many features a selector keeps.
- **Filter method.** Scores each feature on its own, with a statistical test,
  without training a model. Fast. Examples: ANOVA F, mutual information,
  correlation, variance threshold.
- **Embedded method.** Selection happens as a side effect of training a model.
  Examples: Lasso (weights driven to zero), tree-based importance.
- **Wrapper method.** Trains a model repeatedly on different feature subsets
  and keeps the best. Slow but accounts for features working together.
  Example: recursive feature elimination (RFE).
- **Higher Criticism (HC).** A thresholding rule from Donoho and Jin (2008).
  It takes a p-value per feature and chooses the cut-off itself, so it sets
  its own `k`.
- **Random selector.** Picks `k` features at random. The control every method
  must beat.
- **Model (classifier).** Trained on the selected features to predict `y`.
  Examples: logistic regression, random forest, k-NN.
- **Split.** A division of the samples into a training part and a test part.
- **Monte Carlo splits.** Many independent random splits, each stratified so
  class proportions are kept. Gives a distribution of scores, not one number.
- **Selection inside the split.** The selector is fitted on the training part
  only. Fitting it on all the data first leaks test information and inflates
  scores.
- **Task.** Every task is binary. A target with more than two classes, such
  as PAM50, becomes one task per class: that class vs the rest.
- **Macro-F1.** The F1 score averaged over the two classes with equal weight.
  The main metric, because each class-vs-rest task is imbalanced.
- **Stability.** How similar the chosen features are across splits. Measured
  with Kuncheva's index, where random selection scores 0.
- **Grid.** Every combination of selector, `k`, model and split,
  run under one protocol.
- **Config.** A YAML file that describes an experiment, so it can run from the
  command line.

## The problem

Data either comes from files or is already in memory. Simulation output,
preprocessed data and neural-network embeddings are all the in-memory case:
some earlier step builds `X, y` and holds it in memory. The statistical code
should work the same way for both.

One input is optional: the indices of the truly informative features. Only a
simulation knows them. When given, the experiment also scores whether each
selector found them.

There is one grid runner, `featsel/run.py`, using Monte Carlo splits. Its core
is `run(X, y, config)`, which takes data in memory. `run_config(config)` loads
from files and calls it. Stability and summaries live in `featsel/analysis.py`.

## Design 1: Functions only

No classes beyond the selectors. One core function takes `X, y` and plain
arguments.

```python
from featsel import evaluate, load

# in memory (simulation, preprocessed data, embeddings)
res = evaluate(X, y, selectors=['anova_f', 'hc'], k=[10, 50], models=['logreg'],
               splits=100, train_size=0.7, seed=42, n_jobs=-1)
res = evaluate(X, y, ..., truth=true_idx)       # optional ground truth

# from files
X, y = load('configs/scanb.yaml')
res = evaluate(X, y, **yaml.safe_load(open('configs/exp.yaml')))
```

```bash
featsel run configs/exp.yaml          # = load + evaluate + to_csv
```

- Gain: smallest surface. Trivial to test.
- Cost: settings travel as long keyword lists. There is no object to reuse
  across datasets.

## Design 2: One runner, protocol as a config

The protocol is plain data: a dict, or the same thing read from YAML. Data is
passed in when it runs.

```python
from featsel.run import load_config, run, run_config

config = {
    'selectors': [{'name': 'anova_f', 'k': [10, 50]}, {'name': 'higher_criticism'}],
    'models': [{'name': 'logistic_regression'}],
    'n_splits': 100, 'train_sizes': [0.7], 'seed': 42, 'n_jobs': -1,
    'output': 'results/run.csv',
}
run(X, y, config)                                # in memory
run(X, y, config, truth=true_idx)                # in memory, ground truth known
run_config(load_config('configs/exp.yaml'))      # files; the CLI does this
```

- Gain: one protocol can be applied unchanged to many datasets. That is
  task 4. Python and the command line share one format.
- Cost: a typo in the config surfaces when the run starts, not earlier.

## Design 3: Pure scikit-learn

Everything is a scikit-learn object. `featsel` only adds selectors and a
scorer for results.

```python
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from featsel.selectors import ANOVA, HigherCriticism
from featsel.metrics import SCORERS, stability

pipe = Pipeline([('scale', StandardScaler()), ('sel', ANOVA(k=50)),
                 ('clf', LogisticRegression())])
cv = StratifiedShuffleSplit(n_splits=100, train_size=0.7, random_state=42)
out = cross_validate(pipe, X, y, cv=cv, scoring=SCORERS,
                     return_estimator=True, n_jobs=-1)
stability([e['sel'].support_ for e in out['estimator']])

# the grid over selectors x k x models = GridSearchCV or a hand-written loop
```

- Gain: almost no own code. Parallelism comes free with `n_jobs`. Familiar
  API for any reader.
- Cost: timing selection separately from training, per-split stability and
  resume-after-crash all need workarounds. Task 3 shrinks to "I set
  `n_jobs`", which is thin for the report.

## Design 4: Dataset objects

Each data source is a class with one common interface. The runner only sees
that interface.

```python
class Dataset(Protocol):
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    name: str
    true_features: np.ndarray | None

ds = CSVDataset('configs/scanb.yaml')
ds = ArrayDataset(X, y, name='sim_seed0', true_features=true_idx)

run(ds, config)
```

- Gain: names, ground truth and metadata travel with the data. Results are
  labelled by `ds.name` automatically.
- Cost: in-memory data needs a wrapper (`ArrayDataset`) instead of a bare
  `X, y`. More classes.

## Uses every design must support

A single selector as a scikit-learn step:

```python
from featsel.selectors import HigherCriticismSelector

pipe = make_pipeline(StandardScaler(), HigherCriticismSelector(),
                     LogisticRegression())
pipe.fit(X_train, y_train)
pipe[1].get_support()     # boolean mask
pipe[1].get_support().sum()  # HC chooses its own count
```

Transfer learning with embeddings already in memory:

```python
from featsel.selectors import ANOVAFSelector

emb = backbone(images).detach().numpy()
sel = ANOVAFSelector(n_features=64).fit(emb, y)
head = nn.Linear(64, n_classes)
logits = head(backbone(batch)[:, sel.get_support(indices=True)])
```

A new selector registers itself by being defined, in its own subpackage
`featsel/selectors/corr_rank/`:

```python
class CorrelationRankSelector(FeatureSelector, aliases='corr_rank'):
    def fit(self, X, y):
        scores = np.abs([np.corrcoef(col, y)[0, 1] for col in X.T])
        self.mask_ = np.zeros(X.shape[1], dtype=bool)
        self.mask_[np.argsort(scores)[-self.n_features:]] = True
        return self

    def _get_support_mask(self):
        return self.mask_
```

## Comparison

| Question | 1 Functions | 2 Config runner | 3 sklearn | 4 Dataset objects |
|---|---|---|---|---|
| Bare `X, y` in memory | yes | yes | yes | needs a wrapper |
| Same protocol on many datasets | pass args again | reuse the config | reuse `pipe` + `cv` | reuse the config |
| Optional ground truth | extra argument | extra argument | manual | field on the dataset |
| Timing selection separately | yes | yes | workaround | yes |
| Task 3 has real content | yes | yes | thin | yes |
| Code to write | least | little | least | most |

## Decision

Design 2, decided 2026-09-26. It is what `featsel/run.py` already is.

- The protocol is a config: a dict in Python, a YAML file on the command
  line. No class wraps it, because the dict already carries everything.
- `run(X, y, config, truth=None)` is the core. Data always arrives as `X, y`
  in memory.
- `run_config(config)` loads the dataset named in the config and calls `run`.
  The command line calls `run_config`.
- Paths in a config file are relative to that file, so the package never
  assumes this repo's folder layout.
- Parallelism lives in `run` only: `n_jobs` workers, one split per task.

Rejected:

- Design 1: the same as Design 2 but with the config spread over keyword
  arguments, so the YAML and Python forms would differ.
- Design 3: timing selection separately, stability and resume all need
  workarounds, and task 3 would shrink to setting `n_jobs`.
- Design 4: in-memory data and simulations are the same case, and ground
  truth fits as one argument. A dataset class would only wrap `X, y`.

Limits: the config is not validated before the run starts. A typo fails at
the first split that uses it.

## Selector design

Decided 2026-09-26, extended 2026-09-28. Built 2026-09-29. RFE changed
2026-09-30.

**Status.** Built as described below. The old wrapper
(`featsel/feature_selector.py`) is removed. The runner and the configs name
selectors by class name. Not built: Higher Criticism.

**Problem (before this design).** There were two layers. The selector classes in `selectors/`
do the work but are not scikit-learn estimators, so they cannot go into a
scikit-learn `Pipeline` directly. The `FeatureSelector` wrapper picks a class
by name and forwards parameters through a hand-written whitelist. A parameter
missing from the whitelist is silently dropped: a typo in a config runs with
the default and nobody notices. Adding a selector means editing three places.

**Decision.**

- `FeatureSelector` becomes the base class of every selector. It inherits
  scikit-learn's `BaseEstimator` and `SelectorMixin`. Each subclass lists its
  parameters in `__init__`, only stores them, and writes `fit`, which sets
  `support_`. `transform`, `get_support` and `get_feature_names_out` come
  from `SelectorMixin`; `get_params`, `set_params` and `clone` from
  `BaseEstimator`.
- As in scikit-learn, `transform` returns a NumPy array. Pandas output is
  opt-in with `set_output(transform='pandas')`. The old base class returned
  a DataFrame for DataFrame input.
- One instance handles one selection method.
- Every subclass registers itself automatically, through
  `__init_subclass__`, under its class name. It may add aliases as a class
  keyword, `aliases: None | str | list[str]`, for example
  `class MySelector(FeatureSelector, aliases=['mine', 'my_sel'])`. A
  keyword is not inherited, so a subclass never re-registers its parent's
  aliases. An alias equal to the class name is ignored. Skipping abstract
  classes was built and removed 2026-10-31: no selector had an abstract
  parent.
- Names are stored lowercased in one dict, so lookup is case-insensitive. Two
  classes claiming the same lowercased name raise an error when the second is
  defined.
- In Python, users import the class and make an instance:
  `ANOVAFSelector(n_features=20)`. A misspelled class name fails at import.
- `FeatureSelector.create(name, **params)` is only for names that arrive as
  text, such as a YAML config read by the runner. It looks the class up and
  passes the parameters through untouched.
- Results always record the class name, whatever alias a config used, so runs
  stay comparable.
- A selector whose ranking, cut at any `k`, equals a fit at that `k` sets the
  class flag `rank_once` and gives the order through `ranking()`. The runner
  then fits once per split. Added 2026-10-31. Before, the runner kept these
  facts as lists of class names.
- A selector that uses randomness takes `random_state`, default 42, so a fit
  repeats without a seed. It draws from a generator built afresh on each fit.
  A generator kept in the instance was rejected: scikit-learn's `clone` does
  not copy it, and its draws would depend on the order of fits. Added
  2026-11-01.
- How a selector computes its result is its own business. It may wrap
  scikit-learn, or be its own CPU or GPU implementation. Replacing one with
  another keeps the class name and parameters, so configs and results do not
  change.
- Wrappers by default: ANOVA F, FDR, FWE, percentile, mutual information,
  variance, correlation, Lasso, tree importance, RFE, RFECV, sequential
  selection. Own implementation: Higher Criticism, which scikit-learn lacks.
  Any other selector gets its own implementation only if a timing run shows a
  real gain, checked against scikit-learn's selection.
- Each wrapper is tested against the scikit-learn method it wraps: the same
  selected features, and `fit` plus `get_support` at most 5% slower.
- RFE, changed 2026-09-30. It was planned as an own implementation, because
  it dominates cost. It wraps scikit-learn's `RFE` for now, to get the full
  pipeline running end to end. On its own, a hand-written loop does the same
  work as scikit-learn's. The saving comes from ranking features once per
  split and reading every `k` from that one ranking, which is a change to the
  runner. An own implementation comes after that.

**Layout.** One subpackage per selector, not per family:

```
featsel/selectors/
├── __init__.py          # imports every selector, which registers it
├── base.py
└── anova_f/
    ├── __init__.py      # the class; more files beside it as it grows
    ├── tests/
    │   └── test_anova_f.py
    └── demos/           # notebooks, when one is worth writing
```

A family (filter, embedded, wrapper) is a label, not shared code, so it gets
no folder. The random selector lives in `random_selection/`: a folder named
`random` would shadow the standard library module.

**Removed.** The old wrapper, its parameter whitelist, and the union,
intersection and voting modes for combining methods.

**Rejected.**

- Keeping the wrapper: the silent-drop defect stays.
- A hand-written name-to-class dict: explicit, but every selector needs a
  second edit.
- A short declared name per class (`name = 'anova_f'`) instead of the class
  name: it only protects against class renames, which aliases also cover.
- Reimplementing every scikit-learn algorithm: most selectors cost seconds
  per split while model fitting costs minutes, so rewriting them gains
  nothing measurable.
- Wrapping RFE for good: RFE is the one selector whose cost makes the full
  design impractical. Wrapping it is accepted only until the runner ranks
  once per split (see RFE above).
- Family files or family folders: a file per family grows unmanageable once
  a selector has CPU and GPU versions, and a family folder adds nesting
  without shared code.
- `create` as the Python entry point: a misspelled name fails only at run
  time, where an import fails at once.

**Limits.**

- The class name is a public identifier. It appears in configs and results
  files. Renaming a class needs an alias for configs. Old results files keep
  the old name.
- Aliases are added only when a rename actually happens.
- A class registers only when its module is imported. `featsel/__init__.py`
  imports `featsel.selectors`, which imports every selector.
- Results files written before the switch to class names use the old method
  names (`random`, `anova_f`). They are not comparable by name with new runs,
  and `--resume` on them recomputes everything. Notebook 02 names its control
  `random`, matching the current `results/experiment_classifiers.csv`; it
  needs `RandomSelector` after that file is regenerated.
- Validation happens in `fit`, as scikit-learn requires, so a missing
  required `n_features` fails at fit time, not at construction.
- Selector tests live inside the package. Packaging excludes them, and pytest
  searches both `tests/` and `featsel/`.

## Dataset loaders

Decided 2026-10-16. Built for SCAN-B as `featsel_datasets/scanb_gse96058/`,
and on 2026-10-17 for BreastMNIST as `featsel_datasets/medmnist_breastmnist/`. There,
the generic class also computes ResNet-50 embeddings once and caches them;
it imports PyTorch only inside that step, so the `featsel` package and
cached runs never need it.

**Problem.** The SCAN-B files came from the course with no recorded source or
filters. Their origin, which patients they hold and how genes were chosen
could not be checked. Other researchers on the same data could not reuse our
preparation.

**Decision.** Each dataset gets a folder in `featsel_datasets/<name>/` with
two layers.

- A generic class (`gse96058.py`). It downloads the published files once,
  keeps them in the folder as a cache, and exposes them unchanged: every
  sample, every gene, every patient field under its published name. It makes
  no choice for the caller. Other researchers can use it alone.
- A function holding this project's choices (`featsel_view.py`, `load_xy`).
  For SCAN-B: HiSeq samples only, one sample per patient, the chosen label,
  and no single-value genes. It returns `X, y` for `run`. It is a function,
  not a class, because it holds no state.
- `X` is a DataFrame and `y` a Series. Gene names reach the selectors, and
  patients cannot be misaligned with their labels. `X.to_numpy()` gives an
  array when needed.
- `featsel_view.py` also defines `NAME` and `TARGET`. An experiment config
  names the module as its `dataset`; the runner imports it and calls
  `load_xy(target=...)`. There is no dataset YAML (decided 2026-10-17). The
  module owns the data choices; the experiment config owns the protocol and
  the `k` values, which are feature counts for that dataset.
- A notebook in `demos/` shows the dataset's content and traps.
- Git ignores data files by type (`*.csv`, `*.gz`, `*.npz`, `*.txt`
  anywhere), not the folder, so the code is tracked. A file that must be
  tracked is named as an exception.

**Rejected.**

- A loader class per dataset inside the `featsel` package: the package stays
  generic; dataset code lives beside the data.
- Returning NumPy arrays and a name list: alignment by position is what the
  index protects against. GSE96058's gene file and patient file list samples
  in different orders.
- A second class for the project's choices: it would hold no state.
- Naming the folder `datasets/`: the Hugging Face `datasets` library would
  shadow it on import.
- A dataset YAML naming the loader (`loader: module:function` plus
  `params`), used briefly on 2026-10-17: it repeated the choices `load_xy`
  already makes, so the two could drift apart.
- Moving `k` and the output name into `featsel_view.py` too: the experiment
  config would no longer show the full experiment.

**Limits.**

- The runner imports a module named in a config. This names code from a
  config, which is close to the extension points `SCOPE.md` section 4 rules
  out. It is one convention (`NAME`, `TARGET`, `load_xy`), with no registry.
- The runner tells a module from a CSV config by the `.yaml`/`.yml` ending.
- `featsel_datasets` is a regular package, with an `__init__.py` per folder,
  so each dataset's `featsel_view` has one module name. It is excluded from
  the installed `featsel` package; code outside the repo root must put the
  root on `sys.path`.
- Dataset folders are named after source and dataset (`scanb_gse96058`,
  `medmnist_breastmnist`); the generic module is named after the dataset
  (`gse96058.py`, `breastmnist.py`).
- The first load downloads 592 MB and takes about 85 s.
