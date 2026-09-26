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
- **Experiment / grid.** Every combination of selector, `k`, model and split,
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

Today there are two grid runners with different split protocols
(`experiment.run_grid` uses k-fold, `run.run` uses Monte Carlo splits). Only
the second is config-driven, and it reads data from disk only.

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

## Design 2: An `Experiment` object

The protocol is an object. Data is passed in when it runs.

```python
exp = Experiment(selectors=['anova_f', 'hc'], k=[10, 50], models=['logreg'],
                 splits=MonteCarlo(n=100, train_size=0.7), seed=42)

exp.run(X, y, n_jobs=-1)                         # in memory
exp.run(X, y, truth=true_idx)                    # in memory, ground truth known
exp.run(*load('configs/scanb.yaml'))             # files
Experiment.from_yaml('exp.yaml').run(...)        # CLI underneath
```

- Gain: one protocol can be applied unchanged to many datasets. That is
  task 4.
- Cost: one more class to design and document.

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

Each data source is a class with one common interface. The experiment only
sees that interface.

```python
class Dataset(Protocol):
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    name: str
    true_features: np.ndarray | None

ds = CSVDataset('configs/scanb.yaml')
ds = ArrayDataset(X, y, name='sim_seed0', true_features=true_idx)

Experiment(...).run(ds, n_jobs=-1)
```

- Gain: names, ground truth and metadata travel with the data. Results are
  labelled by `ds.name` automatically.
- Cost: in-memory data needs a wrapper (`ArrayDataset`) instead of a bare
  `X, y`. More classes.

## Uses every design must support

A single selector as a scikit-learn step:

```python
from featsel.selectors import HigherCriticism

pipe = make_pipeline(StandardScaler(), HigherCriticism(test='anova'),
                     LogisticRegression())
pipe.fit(X_train, y_train)
pipe[1].support_          # boolean mask
pipe[1].n_selected_       # HC chooses its own k
```

Transfer learning with embeddings already in memory:

```python
emb = backbone(images).detach().numpy()
sel = ANOVA(k=64).fit(emb, y)
head = nn.Linear(64, n_classes)
logits = head(backbone(batch)[:, sel.indices_])
```

A custom selector, with no registration:

```python
class MySelector(BaseSelector):
    def _score(self, X, y):
        return np.abs(np.corrcoef(X.T, y)[-1, :-1])
```

## Comparison

| Question | 1 Functions | 2 Experiment | 3 sklearn | 4 Dataset objects |
|---|---|---|---|---|
| Bare `X, y` in memory | yes | yes | yes | needs a wrapper |
| Same protocol on many datasets | pass args again | reuse the object | reuse `pipe` + `cv` | reuse the object |
| Optional ground truth | extra argument | extra argument | manual | field on the dataset |
| Timing selection separately | yes | yes | workaround | yes |
| Task 3 has real content | yes | yes | thin | yes |
| Code to write | least | medium | least | most |

Designs 2 and 4 can be combined: `run` accepts either `X, y` or a `Dataset`.

## Decision

Open.
