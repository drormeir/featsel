# Configuration Files

Dataset-specific configuration files for the feature selection pipeline.

## Files

| File | Description |
|------|-------------|
| `template.yaml` | Documented template with all available options |
| `scanb.yaml` | SCAN-B breast cancer dataset configuration |

## Task Types

The pipeline supports three task types:

| Task | Config Setting | Use Case |
|------|----------------|----------|
| Binary classification | `type: classification`, `classification_type: binary` | Two classes (e.g., disease vs healthy) |
| Multi-class classification | `type: classification`, `classification_type: multiclass` | Three or more classes (e.g., cancer subtypes) |
| Regression | `type: regression` | Continuous target (e.g., survival time, gene expression level) |

## Creating a New Config

1. Copy `template.yaml` to `<dataset_name>.yaml`
2. Fill in required fields (marked `[REQUIRED]` in template)
3. Adjust optional settings as needed
4. Run: `python -m src.run --config configs/<dataset_name>.yaml`

## Feature Selection Methods by Task Type

### Classification
- Variance threshold
- Chi-squared test
- ANOVA F-test (`f_classif`)
- Mutual information (`mutual_info_classif`)
- L1-regularized logistic regression
- Tree-based importance (Random Forest, Gradient Boosting)

### Regression
- Variance threshold
- Pearson/Spearman correlation
- F-test (`f_regression`)
- Mutual information (`mutual_info_regression`)
- Lasso (L1-regularized linear regression)
- Tree-based importance (Random Forest, Gradient Boosting)
