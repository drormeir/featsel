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

- **Chapter 2 - Background**: Theoretical foundations of feature selection methods and their applicability to high-dimensional data
- **Chapter 3 - Methods**: Detailed description of the implemented feature selection pipeline and evaluation methodology
- **Chapter 4 - Experiments**: Experimental setup, parameter configurations, and execution details
- **Chapter 5 - Results**: Comparative analysis of feature selection methods and model performance
- **Chapter 6 - Conclusion**: Summary of findings, limitations, and future directions
