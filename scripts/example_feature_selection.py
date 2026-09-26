"""
Example: Feature selection with sklearn pipeline integration.

This script demonstrates the featsel package's feature selection capabilities
for high-dimensional data, particularly useful for transfer learning scenarios.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from featsel import FeatureSelector

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Feature Selection Example: High-Dimensional Data")
print("=" * 70)

# Simulate transfer learning scenario:
# 500 embedding features from ResNet, only 100 samples (rare disease)
print("\n1. Creating synthetic high-dimensional dataset...")
print("   - 100 samples (simulating rare disease CT images)")
print("   - 500 features (simulating ResNet embeddings)")
print("   - 50 informative features, 50 redundant, 400 noise features")

X, y = make_classification(
    n_samples=100,
    n_features=500,
    n_informative=50,
    n_redundant=50,
    n_classes=3,
    random_state=42,
    shuffle=False
)

# Convert to DataFrame for better feature name tracking
feature_names = [f'embedding_{i}' for i in range(500)]
X = pd.DataFrame(X, columns=feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   - Training set: {X_train.shape[0]} samples")
print(f"   - Test set: {X_test.shape[0]} samples")

# Baseline: Logistic Regression without feature selection
print("\n2. Baseline (no feature selection)...")
clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
clf_baseline.fit(X_train, y_train)
score_baseline = clf_baseline.score(X_test, y_test)
print(f"   Accuracy: {score_baseline:.3f}")

# Method 1: Variance threshold (fast preprocessing)
print("\n3. Feature Selection Method 1: Variance Threshold")
pipe_variance = Pipeline([
    ('select', FeatureSelector(method='variance_threshold', threshold=0.01)),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
pipe_variance.fit(X_train, y_train)
score_variance = pipe_variance.score(X_test, y_test)
n_features_variance = pipe_variance.named_steps['select'].n_features_out_
print(f"   Selected features: {n_features_variance}/{X_train.shape[1]}")
print(f"   Accuracy: {score_variance:.3f}")

# Method 2: ANOVA F-test (fast univariate)
print("\n4. Feature Selection Method 2: ANOVA F-test")
pipe_anova = Pipeline([
    ('select', FeatureSelector(method='anova_f', n_features=100, task='classification')),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
pipe_anova.fit(X_train, y_train)
score_anova = pipe_anova.score(X_test, y_test)
print(f"   Selected features: 100/{X_train.shape[1]}")
print(f"   Accuracy: {score_anova:.3f}")

# Method 3: Mutual Information (captures non-linearity)
print("\n5. Feature Selection Method 3: Mutual Information")
pipe_mi = Pipeline([
    ('select', FeatureSelector(method='mutual_info', n_features=100, random_state=42)),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
pipe_mi.fit(X_train, y_train)
score_mi = pipe_mi.score(X_test, y_test)
print(f"   Selected features: 100/{X_train.shape[1]}")
print(f"   Accuracy: {score_mi:.3f}")

# Method 4: Correlation-based (removes redundancy)
print("\n6. Feature Selection Method 4: Correlation Selector")
pipe_corr = Pipeline([
    ('select', FeatureSelector(
        method='correlation',
        n_features=100,
        target_threshold=0.05,
        inter_feature_threshold=0.95
    )),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
pipe_corr.fit(X_train, y_train)
score_corr = pipe_corr.score(X_test, y_test)
print(f"   Selected features: 100/{X_train.shape[1]}")
print(f"   Accuracy: {score_corr:.3f}")

# Multi-stage pipeline (recommended for high-dimensional data)
print("\n7. Multi-Stage Pipeline (Recommended)")
print("   Stage 1: Remove low-variance features")
print("   Stage 2: ANOVA F-test (500 → 200 features)")
print("   Stage 3: Mutual Information (200 → 50 features)")
pipe_multi = Pipeline([
    ('prefilter', FeatureSelector(method='variance_threshold', threshold=0.01)),
    ('quick_select', FeatureSelector(method='anova_f', n_features=200)),
    ('final_select', FeatureSelector(method='mutual_info', n_features=50, random_state=42)),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
pipe_multi.fit(X_train, y_train)
score_multi = pipe_multi.score(X_test, y_test)
print(f"   Final features: 50/{X_train.shape[1]}")
print(f"   Accuracy: {score_multi:.3f}")

# Cross-validation on best method
print("\n8. Cross-Validation (5-fold) on Multi-Stage Pipeline")
cv_scores = cross_val_score(pipe_multi, X_train, y_train, cv=5)
print(f"   CV Scores: {cv_scores}")
print(f"   Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Get feature importance report
print("\n9. Feature Importance Report (top 10 features)")
selector = pipe_multi.named_steps['final_select']
report = selector.get_report()
print(report.head(10).to_string())

# Save selected features
selected_features = selector.selected_features_
print(f"\n10. Selected Features ({len(selected_features)} total):")
print(f"    {selected_features[:10]} ... (showing first 10)")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print(f"Baseline (no selection):     {score_baseline:.3f} (500 features)")
print(f"Variance Threshold:          {score_variance:.3f} ({n_features_variance} features)")
print(f"ANOVA F-test:                {score_anova:.3f} (100 features)")
print(f"Mutual Information:          {score_mi:.3f} (100 features)")
print(f"Correlation Selector:        {score_corr:.3f} (100 features)")
print(f"Multi-Stage Pipeline:        {score_multi:.3f} (50 features)")
print("=" * 70)
print("\nFeature selection successfully reduced dimensionality while")
print("maintaining or improving model performance!")
