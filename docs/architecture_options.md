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
- **Macro-F1.** The F1 score averaged over classes with equal weight. The main
  metric, because the classes are imbalanced.
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
emb = backbone(images).detach().numpy()
sel = FeatureSelector.create('ANOVAFSelector', n_features=64).fit(emb, y)
head = nn.Linear(64, n_classes)
logits = head(backbone(batch)[:, sel.get_support(indices=True)])
```

A custom selector registers itself by being defined:

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

Decided 2026-09-26. Not yet implemented.

**Problem.** Today there are two layers. The selector classes in `selectors/`
do the work but are not scikit-learn estimators, so they cannot go into a
scikit-learn `Pipeline` directly. The `FeatureSelector` wrapper picks a class
by name and forwards parameters through a hand-written whitelist. A parameter
missing from the whitelist is silently dropped: a typo in a config runs with
the default and nobody notices. Adding a selector means editing three places.

**Decision.**

- `FeatureSelector` becomes the base class of every selector. It inherits
  scikit-learn's `BaseEstimator` and `SelectorMixin`. Each subclass writes
  `fit` and `_get_support_mask`; `transform`, `get_support` and
  `get_feature_names_out` come from `SelectorMixin`.
- One instance handles one selection method.
- Every concrete subclass registers itself automatically, through
  `__init_subclass__`, under its class name. It may add aliases as a class
  keyword, `aliases: None | str | list[str]`, for example
  `class ANOVAFSelector(FeatureSelector, aliases=['anova', 'anova_f'])`. A
  keyword is not inherited, so a subclass never re-registers its parent's
  aliases. An alias equal to the class name is ignored. Abstract classes do
  not register.
- Names are stored lowercased in one dict, so lookup is case-insensitive. Two
  classes claiming the same lowercased name raise an error when the second is
  defined.
- `FeatureSelector.create(name, **params)` looks the class up and passes the
  parameters through untouched, so a typo fails in the constructor.
- Results always record the class name, whatever alias a config used, so runs
  stay comparable.
- Selectors call scikit-learn's implementations where they exist (`f_classif`,
  `mutual_info_classif`, `RFE`, `SelectFromModel`). The package adds what
  scikit-learn lacks: Higher Criticism and the random control.

**Removed.** The old wrapper, its parameter whitelist, and the union,
intersection and voting modes for combining methods.

**Rejected.**

- Keeping the wrapper: the silent-drop defect stays.
- A hand-written name-to-class dict: explicit, but every selector needs a
  second edit.
- A short declared name per class (`name = 'anova_f'`) instead of the class
  name: it only protects against class renames, which aliases also cover.
- Reimplementing scikit-learn's algorithms: more code to test, and nothing
  gained over a widely used implementation.

**Limits.**

- The class name is a public identifier. It appears in configs and results
  files. Renaming a class needs an alias for configs. Old results files keep
  the old name.
- A class registers only when its module is imported, so `create` must make
  sure `featsel.selectors` is loaded.
- Aliases are added only when a rename actually happens.
