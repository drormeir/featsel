# Conclusion

Draft.

## What was built

The project delivers `featsel`, a feature selection package for
high-dimensional data. Its selectors cover the filter, embedded and wrapper
families and Higher Criticism. Each one is a scikit-learn estimator, so it
drops into an existing pipeline. One config file drives a full experiment:
`python -m featsel.run --config ...`. The runner selects features inside
every split, tunes the classifier inside every training half, compares
every method with a random control, and writes one results table.

The runner is faster than plain scikit-learn on the same grid, with
identical scores. Most of the saving comes from preparing each split once
and ranking features once for all `k`. Parallel workers add to that.

The pipeline ran on two datasets: SCAN-B gene expression and BreastMNIST
image embeddings.

## What was learned

- Feature selection pays most when few features are allowed. As the budget
  grows, random features catch up, because the signal is spread over many
  correlated features.
- Mann-Whitney, Lasso and tree importance form a top group. Five splits
  cannot rank them. The simple U-test filter is the most stable of them.
- RFE with a coarse step is the weakest selector on SCAN-B.
- Higher Criticism thresholding reduced to a top-k filter on both datasets.
  Its assumptions, rare signal and independent features, do not hold there.
- Good prediction does not need the genes that define the label. Many
  feature sets predict equally well, so a selected list is not a unique
  answer.
- The main conclusions held on both datasets. The fine ranking did not.

## Future work

- More splits, to narrow the ranges enough to rank the top group.
- More values of `k`, to find where each method's gain over random
  selection ends.
- A finer RFE step, and tuned selector settings.
- Other classifiers, to check that the ranking does not depend on logistic
  regression.
- A dataset with rare, weak signal, where Higher Criticism is expected to
  choose its own count.
- A measured clinical target on SCAN-B, such as pathologist ER status, in
  place of computed PAM50 labels.
