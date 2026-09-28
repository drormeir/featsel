# Introduction

## Background

Feature selection is a fundamental step in machine learning that directly impacts model performance, interpretability, and computational efficiency. The process involves identifying the subset of features that contribute most significantly to the prediction task while discarding irrelevant or redundant variables. In datasets with many features, proper feature selection can mean the difference between a model that generalizes well and one that overfits to noise.

The challenge of feature selection becomes particularly acute in high-dimensional settings, where the number of features approaches or exceeds the number of samples. This scenario, often referred to as the "curse of dimensionality," is common in genomics, text analysis, and other domains where data collection yields thousands of measurements per observation.

## Gene Expression Data

Gene expression profiling measures the activity levels of thousands of genes simultaneously, providing a molecular snapshot of cellular function. Since genes encode proteins that dictate cell behavior, expression patterns can reveal disease states, treatment responses, and biological subtypes.

From a machine learning perspective, gene expression datasets present several challenges:

- **High dimensionality**: Datasets typically contain 10,000 to 50,000 gene features
- **Limited samples**: Clinical studies often have only hundreds of patient samples
- **Feature correlation**: Genes operate in pathways and networks, creating complex correlation structures
- **Biological noise**: Technical and biological variability introduce measurement uncertainty

These characteristics make gene expression data an ideal testbed for developing and evaluating feature selection methods.

## Project Objectives

This project develops a comprehensive feature selection pipeline for high-dimensional data, with a focus on gene expression classification tasks. The primary objectives are:

1. **Implement a feature selection pipeline** capable of handling high-dimensional datasets with multiple selection methods
2. **Compare feature selection techniques** including filter methods, wrapper methods, and embedded approaches
3. **Build scalable infrastructure** with parallelization to enable efficient processing of large datasets
4. **Validate generalization** by testing the pipeline on a secondary dataset from a different domain

## Dataset

The primary dataset used in this project is the SCAN-B breast cancer gene expression dataset. This dataset contains:

- Gene expression measurements across thousands of genes
- Clinical annotations including molecular subtypes (PAM50 classification: Basal, Luminal A, Luminal B, HER2-enriched, and Normal-like)
- Estrogen receptor (ER) status
- Survival outcomes

The PAM50 molecular subtype classification serves as the primary prediction target, representing a clinically relevant multi-class classification problem.

## Report Structure

This report is organized as follows:

- **Chapter 2 - Related Work**: The filter, embedded and wrapper families, Higher Criticism, and prior comparative studies on gene expression
- **Chapter 3 - Data**: SCAN-B and the second dataset, their targets and class imbalance
- **Chapter 4 - Methodology**: The selection methods, the evaluation protocol and the metrics
- **Chapter 5 - Software and Implementation**: The `featsel` package and its parallel infrastructure
- **Chapter 6 - Results**: Comparison of the selection methods across models, targets and datasets
- **Chapter 7 - Discussion**: What the results mean, threats to validity and limitations
- **Chapter 8 - Conclusion**: What was built, what was learned, and future work
