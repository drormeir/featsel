# Appendix C. Full tables

[Back to contents](README.md)

The full output behind chapters 7 and 8. Every AUC is a mean over the 16
splits. `scripts/summarize_frontier.py` wrote the CSVs these tables are
built from, in [`figures/`](figures/): `summary_<run>_curve.csv` (test AUC
at every `k`) and `summary_<run>_chosen.csv` (each method at the `k` it
chose on validation sets). Appendix A gives the commands.

In the chosen-`k` tables, "Selection" is the median time of one fit.
"Search or validation" is the median time per split to choose Lasso's C or
RFECV's count, or to choose `k` on the 5 validation sets. Stability is
Nogueira's measure over the splits (section 3.6). It is undefined for all
features, which every split keeps.

Random selection at every feature is the all-features reference; in the
curve tables it is the last column of the random row.

## C.1 SCAN-B

### C.1.1 SCAN-B: Basal, test AUC at every `k`

| Method | 5 | 10 | 50 | 250 | 500 | 1,000 | 1,500 | 2,000 | 28,541 |
|---|---|---|---|---|---|---|---|---|---|
| Correlation (Spearman) | — | 0.993 | 0.992 | 0.991 | 0.992 | 0.992 | 0.992 | 0.992 | — |
| Mann-Whitney | — | 0.993 | 0.992 | 0.992 | 0.993 | 0.993 | 0.993 | 0.992 | — |
| Mann-Whitney 1000, then RFE | — | 0.990 | 0.988 | 0.991 | 0.992 | — | — | — | — |
| Mann-Whitney 1000, then tree importance | — | 0.992 | 0.992 | 0.994 | 0.994 | — | — | — | — |
| Mutual information | — | 0.992 | 0.993 | 0.992 | 0.993 | 0.994 | 0.993 | 0.992 | — |
| RFE | — | 0.988 | 0.986 | 0.991 | 0.993 | 0.993 | 0.993 | 0.993 | — |
| Random | — | 0.871 | 0.981 | 0.989 | 0.991 | 0.992 | 0.992 | 0.992 | 0.987 |
| Sequential | 0.990 | 0.989 | — | — | — | — | — | — | — |
| Tree importance | — | 0.992 | 0.992 | 0.992 | 0.993 | 0.994 | 0.993 | 0.992 | — |

### C.1.2 SCAN-B: Basal, at the chosen `k`

| Method | Test AUC [Q1, Q3] | Kept: median (min to max) | Selection, s | Search or validation, s | Stability |
|---|---|---|---|---|---|
| Mann-Whitney 1000, then Lasso | 0.994 [0.992, 0.996] | 49 (9 to 127) | 0.78 | 3.44 | 0.34 |
| Lasso | 0.993 [0.993, 0.995] | 84 (67 to 96) | 0.49 | 8.25 | 0.45 |
| Mann-Whitney 1000, then tree importance | 0.993 [0.992, 0.995] | 250 (10 to 500) | 2.65 | 13 | 0.47 |
| RFE | 0.993 [0.991, 0.995] | 1,500 (500 to 2,000) | 7.23 | 29 | 0.44 |
| Mutual information | 0.992 [0.989, 0.995] | 500 (10 to 2,000) | 3.62 | 34 | 0.50 |
| Tree importance | 0.991 [0.989, 0.994] | 1,000 (10 to 2,000) | 15 | 62 | 0.23 |
| Correlation (Spearman) | 0.991 [0.989, 0.994] | 1,500 (10 to 2,000) | 4.86 | 38 | 0.56 |
| Mann-Whitney | 0.991 [0.989, 0.994] | 750 (10 to 2,000) | 0.67 | 21 | 0.46 |
| Mann-Whitney 1000, then RFE | 0.991 [0.988, 0.995] | 375 (10 to 500) | 2.89 | 11 | 0.43 |
| RFECV, by AUC | 0.991 [0.989, 0.994] | 5,709 (2,855 to 19,979) | 7.56 | 43 | 0.29 |
| Berk-Jones | 0.990 [0.986, 0.993] | 2,854 (2,841 to 2,854) | 0.66 | — | 0.93 |
| HC2008 | 0.990 [0.987, 0.993] | 2,854 | 0.67 | — | 0.93 |
| All features | 0.987 [0.982, 0.991] | 28,541 | 0.03 | — | — |
| RFECV, by accuracy | 0.984 [0.989, 0.994] | 2,855 (1 to 2,855) | 6.88 | 38 | 0.47 |
| HC2004 | 0.873 [0.864, 0.886] | 1 | 0.68 | — | 1.00 |

### C.1.3 SCAN-B: Her2, test AUC at every `k`

| Method | 5 | 10 | 50 | 250 | 500 | 1,000 | 1,500 | 2,000 | 28,541 |
|---|---|---|---|---|---|---|---|---|---|
| Correlation (Spearman) | — | 0.941 | 0.972 | 0.982 | 0.983 | 0.985 | 0.987 | 0.986 | — |
| Mann-Whitney | — | 0.941 | 0.972 | 0.981 | 0.983 | 0.985 | 0.987 | 0.986 | — |
| Mann-Whitney 1000, then RFE | — | 0.959 | 0.973 | 0.980 | 0.983 | — | — | — | — |
| Mann-Whitney 1000, then tree importance | — | 0.948 | 0.974 | 0.981 | 0.984 | — | — | — | — |
| Mutual information | — | 0.951 | 0.974 | 0.982 | 0.984 | 0.985 | 0.985 | 0.985 | — |
| RFE | — | 0.874 | 0.937 | 0.966 | 0.975 | 0.977 | 0.980 | 0.980 | — |
| Random | — | 0.790 | 0.902 | 0.938 | 0.963 | 0.973 | 0.975 | 0.978 | 0.981 |
| Sequential | 0.944 | 0.944 | — | — | — | — | — | — | — |
| Tree importance | — | 0.944 | 0.970 | 0.978 | 0.982 | 0.983 | 0.984 | 0.984 | — |

### C.1.4 SCAN-B: Her2, at the chosen `k`

| Method | Test AUC [Q1, Q3] | Kept: median (min to max) | Selection, s | Search or validation, s | Stability |
|---|---|---|---|---|---|
| Mann-Whitney | 0.986 [0.984, 0.988] | 1,750 (250 to 2,000) | 0.67 | 20 | 0.75 |
| Correlation (Spearman) | 0.986 [0.984, 0.989] | 1,500 (250 to 2,000) | 2.88 | 30 | 0.74 |
| Berk-Jones | 0.985 [0.983, 0.987] | 2,854 (2,827 to 2,854) | 0.66 | — | 0.90 |
| HC2008 | 0.985 [0.983, 0.987] | 2,854 | 0.68 | — | 0.90 |
| Mutual information | 0.984 [0.982, 0.987] | 1,250 (500 to 2,000) | 3.59 | 35 | 0.68 |
| Mann-Whitney 1000, then tree importance | 0.984 [0.981, 0.987] | 500 (250 to 500) | 2.97 | 15 | 0.58 |
| Tree importance | 0.983 [0.981, 0.984] | 1,000 (250 to 2,000) | 20 | 72 | 0.28 |
| Mann-Whitney 1000, then RFE | 0.983 [0.980, 0.985] | 500 | 3.00 | 12 | 0.51 |
| Mann-Whitney 1000, then Lasso | 0.982 [0.979, 0.985] | 77.5 (67 to 91) | 0.78 | 3.81 | 0.49 |
| All features | 0.981 [0.977, 0.985] | 28,541 | 0.04 | — | — |
| Lasso | 0.981 [0.978, 0.985] | 194.5 (172 to 318) | 0.56 | 9.85 | 0.36 |
| RFE | 0.980 [0.977, 0.984] | 2,000 (500 to 2,000) | 7.12 | 28 | 0.32 |
| RFECV, by AUC | 0.979 [0.977, 0.982] | 14,271 (5,709 to 22,833) | 5.15 | 40 | 0.26 |
| HC2004 | 0.865 [0.884, 0.905] | 1 (1 to 2) | 0.67 | — | 0.52 |
| RFECV, by accuracy | 0.583 [0.501, 0.656] | 1 | 7.36 | 36 | 0.10 |

### C.1.5 SCAN-B: LumA, test AUC at every `k`

| Method | 5 | 10 | 50 | 250 | 500 | 1,000 | 1,500 | 2,000 | 28,541 |
|---|---|---|---|---|---|---|---|---|---|
| Correlation (Spearman) | — | 0.893 | 0.909 | 0.944 | 0.940 | 0.951 | 0.956 | 0.958 | — |
| Mann-Whitney | — | 0.893 | 0.908 | 0.944 | 0.940 | 0.951 | 0.956 | 0.958 | — |
| Mann-Whitney 1000, then RFE | — | 0.927 | 0.953 | 0.947 | 0.950 | — | — | — | — |
| Mann-Whitney 1000, then tree importance | — | 0.889 | 0.935 | 0.953 | 0.946 | — | — | — | — |
| Mutual information | — | 0.892 | 0.909 | 0.947 | 0.947 | 0.955 | 0.958 | 0.960 | — |
| RFE | — | 0.666 | 0.874 | 0.931 | 0.945 | 0.953 | 0.955 | 0.956 | — |
| Random | — | 0.750 | 0.870 | 0.920 | 0.911 | 0.927 | 0.935 | 0.939 | 0.958 |
| Sequential | 0.918 | 0.919 | — | — | — | — | — | — | — |
| Tree importance | — | 0.892 | 0.926 | 0.952 | 0.956 | 0.960 | 0.961 | 0.961 | — |

### C.1.6 SCAN-B: LumA, at the chosen `k`

| Method | Test AUC [Q1, Q3] | Kept: median (min to max) | Selection, s | Search or validation, s | Stability |
|---|---|---|---|---|---|
| Tree importance | 0.961 [0.959, 0.963] | 1,500 (1,000 to 2,000) | 17 | 66 | 0.35 |
| HC2008 | 0.961 [0.957, 0.963] | 2,854 | 0.68 | — | 0.92 |
| Berk-Jones | 0.961 [0.957, 0.963] | 2,854 (2,850 to 2,854) | 0.66 | — | 0.92 |
| Mutual information | 0.960 [0.959, 0.963] | 2,000 (1,500 to 2,000) | 3.61 | 49 | 0.80 |
| RFECV, by AUC | 0.959 [0.955, 0.963] | 18,552 (8,563 to 28,541) | 3.68 | 36 | 0.14 |
| RFECV, by accuracy | 0.959 [0.955, 0.962] | 17,125 (5,709 to 25,687) | 3.18 | 29 | 0.16 |
| Mann-Whitney | 0.958 [0.954, 0.961] | 2,000 | 0.66 | 35 | 0.92 |
| Correlation (Spearman) | 0.958 [0.955, 0.961] | 2,000 (1,500 to 2,000) | 4.52 | 52 | 0.91 |
| All features | 0.958 [0.954, 0.961] | 28,541 | 0.07 | — | — |
| Mann-Whitney 1000, then Lasso | 0.956 [0.954, 0.961] | 145 (132 to 158) | 0.82 | 5.47 | 0.53 |
| RFE | 0.956 [0.951, 0.959] | 2,000 (1,500 to 2,000) | 6.09 | 23 | 0.34 |
| Lasso | 0.951 [0.949, 0.956] | 487 (467 to 523) | 0.70 | 13 | 0.32 |
| Mann-Whitney 1000, then RFE | 0.951 [0.948, 0.954] | 375 (50 to 500) | 3.05 | 13 | 0.35 |
| Mann-Whitney 1000, then tree importance | 0.951 [0.946, 0.954] | 250 (250 to 500) | 3.13 | 19 | 0.63 |
| HC2004 | 0.874 [0.869, 0.878] | 1 (1 to 2) | 0.68 | — | 0.70 |

### C.1.7 SCAN-B: LumB, test AUC at every `k`

| Method | 5 | 10 | 50 | 250 | 500 | 1,000 | 1,500 | 2,000 | 28,541 |
|---|---|---|---|---|---|---|---|---|---|
| Correlation (Spearman) | — | 0.912 | 0.953 | 0.958 | 0.963 | 0.972 | 0.970 | 0.970 | — |
| Mann-Whitney | — | 0.912 | 0.953 | 0.958 | 0.963 | 0.972 | 0.970 | 0.970 | — |
| Mann-Whitney 1000, then RFE | — | 0.954 | 0.966 | 0.963 | 0.969 | — | — | — | — |
| Mann-Whitney 1000, then tree importance | — | 0.945 | 0.971 | 0.965 | 0.967 | — | — | — | — |
| Mutual information | — | 0.929 | 0.962 | 0.959 | 0.965 | 0.972 | 0.971 | 0.970 | — |
| RFE | — | 0.762 | 0.903 | 0.926 | 0.938 | 0.947 | 0.948 | 0.948 | — |
| Random | — | 0.753 | 0.879 | 0.922 | 0.917 | 0.937 | 0.944 | 0.949 | 0.948 |
| Sequential | 0.949 | 0.950 | — | — | — | — | — | — | — |
| Tree importance | — | 0.945 | 0.966 | 0.967 | 0.969 | 0.971 | 0.970 | 0.970 | — |

### C.1.8 SCAN-B: LumB, at the chosen `k`

| Method | Test AUC [Q1, Q3] | Kept: median (min to max) | Selection, s | Search or validation, s | Stability |
|---|---|---|---|---|---|
| Correlation (Spearman) | 0.971 [0.968, 0.974] | 1,000 (1,000 to 2,000) | 3.53 | 48 | 0.78 |
| Mann-Whitney 1000, then tree importance | 0.971 [0.968, 0.974] | 50 | 2.52 | 16 | 0.63 |
| Mann-Whitney | 0.971 [0.968, 0.974] | 1,250 (1,000 to 2,000) | 0.67 | 36 | 0.78 |
| Tree importance | 0.971 [0.968, 0.973] | 1,500 (50 to 2,000) | 14 | 70 | 0.30 |
| Mutual information | 0.970 [0.968, 0.974] | 2,000 (1,000 to 2,000) | 3.58 | 47 | 0.78 |
| Mann-Whitney 1000, then Lasso | 0.970 [0.968, 0.972] | 116.5 (110 to 129) | 0.80 | 4.86 | 0.48 |
| HC2008 | 0.969 [0.966, 0.972] | 2,854 (2,854 to 2,855) | 0.67 | — | 0.93 |
| Mann-Whitney 1000, then RFE | 0.968 [0.965, 0.971] | 500 (50 to 500) | 3.10 | 13 | 0.53 |
| Berk-Jones | 0.968 [0.965, 0.972] | 2,854 (2,837 to 2,855) | 0.67 | — | 0.93 |
| Lasso | 0.963 [0.960, 0.966] | 385 (354 to 406) | 0.63 | 12 | 0.31 |
| RFECV, by AUC | 0.949 [0.946, 0.953] | 9,990 (2,855 to 28,541) | 5.84 | 42 | 0.09 |
| All features | 0.948 [0.944, 0.956] | 28,541 | 0.03 | — | — |
| RFE | 0.947 [0.944, 0.952] | 2,000 (1,500 to 2,000) | 7.28 | 27 | 0.28 |
| HC2004 | 0.869 [0.858, 0.875] | 1 (1 to 2) | 0.68 | — | 0.45 |
| RFECV, by accuracy | 0.566 [0.500, 0.547] | 1 | 7.31 | 38 | 0.02 |

### C.1.9 SCAN-B: Normal, test AUC at every `k`

| Method | 5 | 10 | 50 | 250 | 500 | 1,000 | 1,500 | 2,000 | 28,541 |
|---|---|---|---|---|---|---|---|---|---|
| Correlation (Spearman) | — | 0.914 | 0.925 | 0.910 | 0.906 | 0.927 | 0.934 | 0.940 | — |
| Mann-Whitney | — | 0.914 | 0.925 | 0.910 | 0.906 | 0.927 | 0.934 | 0.940 | — |
| Mann-Whitney 1000, then RFE | — | 0.916 | 0.920 | 0.910 | 0.919 | — | — | — | — |
| Mann-Whitney 1000, then tree importance | — | 0.911 | 0.933 | 0.917 | 0.920 | — | — | — | — |
| Mutual information | — | 0.914 | 0.923 | 0.912 | 0.908 | 0.932 | 0.945 | 0.948 | — |
| RFE | — | 0.628 | 0.808 | 0.873 | 0.899 | 0.919 | 0.928 | 0.928 | — |
| Random | — | 0.790 | 0.877 | 0.891 | 0.909 | 0.927 | 0.928 | 0.930 | 0.941 |
| Sequential | 0.900 | 0.895 | — | — | — | — | — | — | — |
| Tree importance | — | 0.912 | 0.923 | 0.927 | 0.931 | 0.939 | 0.941 | 0.940 | — |

### C.1.10 SCAN-B: Normal, at the chosen `k`

| Method | Test AUC [Q1, Q3] | Kept: median (min to max) | Selection, s | Search or validation, s | Stability |
|---|---|---|---|---|---|
| Mutual information | 0.948 [0.947, 0.953] | 2,000 | 3.60 | 40 | 0.78 |
| Berk-Jones | 0.947 [0.942, 0.953] | 2,853 (2,848 to 2,854) | 0.66 | — | 0.89 |
| HC2008 | 0.946 [0.941, 0.954] | 2,854 (2,854 to 2,855) | 0.67 | — | 0.89 |
| All features | 0.941 [0.934, 0.946] | 28,541 | 0.03 | — | — |
| Correlation (Spearman) | 0.940 [0.931, 0.948] | 2,000 (1,500 to 2,000) | 2.22 | 34 | 0.90 |
| Mann-Whitney | 0.939 [0.931, 0.948] | 2,000 (1,500 to 2,000) | 0.67 | 26 | 0.90 |
| Mann-Whitney 1000, then Lasso | 0.938 [0.931, 0.944] | 56.5 (47 to 64) | 0.78 | 4.39 | 0.47 |
| Tree importance | 0.937 [0.928, 0.945] | 1,500 (500 to 2,000) | 21 | 71 | 0.17 |
| RFECV, by AUC | 0.931 [0.920, 0.945] | 5,709 (2,855 to 11,417) | 6.16 | 41 | 0.35 |
| Lasso | 0.930 [0.920, 0.941] | 245.5 (234 to 390) | 0.58 | 10 | 0.33 |
| RFE | 0.928 [0.919, 0.936] | 2,000 (1,500 to 2,000) | 6.54 | 27 | 0.29 |
| Mann-Whitney 1000, then tree importance | 0.927 [0.919, 0.936] | 50 (10 to 500) | 2.78 | 16 | 0.26 |
| Mann-Whitney 1000, then RFE | 0.918 [0.910, 0.923] | 500 (10 to 500) | 3.17 | 13 | 0.32 |
| HC2004 | 0.889 [0.879, 0.900] | 1 (1 to 2) | 0.68 | — | 0.68 |
| RFECV, by accuracy | 0.505 [0.501, 0.507] | 1 | 6.72 | 35 | 0.23 |

## C.2 BreastMNIST

Without the copied images (section 8.8). Sequential selection has not
yet run on them (section 10.1).

### C.2.1 BreastMNIST, test AUC at every `k`

| Method | 10 | 50 | 250 | 500 | 1,000 | 2,048 |
|---|---|---|---|---|---|---|
| Correlation (Spearman) | 0.831 | 0.842 | 0.851 | 0.866 | 0.875 | — |
| Mann-Whitney | 0.831 | 0.842 | 0.851 | 0.866 | 0.883 | — |
| Mann-Whitney 500, then RFE | 0.792 | 0.836 | 0.859 | — | — | — |
| Mann-Whitney 500, then tree importance | 0.831 | 0.863 | 0.858 | — | — | — |
| Mutual information | 0.837 | 0.857 | 0.853 | 0.872 | 0.886 | — |
| RFE | 0.776 | 0.836 | 0.879 | 0.889 | 0.892 | — |
| Random | 0.675 | 0.772 | 0.808 | 0.853 | 0.876 | 0.895 |
| Tree importance | 0.835 | 0.862 | 0.855 | 0.877 | 0.887 | — |

### C.2.2 BreastMNIST, at the chosen `k`

| Method | Test AUC [Q1, Q3] | Kept: median (min to max) | Selection, s | Search or validation, s | Stability |
|---|---|---|---|---|---|
| All features | 0.895 [0.877, 0.906] | 2,048 | 0.00 | — | — |
| RFE | 0.891 [0.875, 0.906] | 1,000 (500 to 1,000) | 0.12 | 0.76 | 0.27 |
| Mutual information | 0.887 [0.873, 0.901] | 1,000 (500 to 1,000) | 0.07 | 0.76 | 0.22 |
| Tree importance | 0.885 [0.872, 0.894] | 1,000 (500 to 1,000) | 0.37 | 2.38 | 0.28 |
| RFECV, by AUC | 0.884 [0.863, 0.902] | 416 (212 to 1,232) | 0.10 | 0.96 | 0.25 |
| Mann-Whitney | 0.883 [0.869, 0.896] | 1,000 (500 to 1,000) | 0.01 | 0.43 | 0.63 |
| Lasso | 0.882 [0.868, 0.898] | 90.5 (73 to 182) | 0.08 | 1.97 | 0.34 |
| Correlation (Spearman) | 0.866 [0.848, 0.881] | 772.5 (10 to 813) | 0.02 | 0.47 | 0.61 |
| Mann-Whitney 500, then RFE | 0.860 [0.837, 0.879] | 250 (50 to 250) | 0.11 | 0.59 | 0.40 |
| Mann-Whitney 500, then Lasso | 0.858 [0.845, 0.865] | 66 (57 to 138) | 0.08 | 2.22 | 0.38 |
| Mann-Whitney 500, then tree importance | 0.855 [0.840, 0.868] | 250 (10 to 250) | 0.40 | 2.15 | 0.46 |
| HC2008 | 0.848 [0.839, 0.867] | 204 | 0.01 | — | 0.77 |
| RFECV, by accuracy | 0.847 [0.816, 0.885] | 212 (8 to 416) | 0.11 | 0.83 | 0.21 |
| Berk-Jones | 0.847 [0.839, 0.866] | 203 (175 to 204) | 0.01 | — | 0.77 |
| HC2004 | 0.771 [0.752, 0.790] | 1 (1 to 2) | 0.01 | — | 0.85 |

## C.3 FGVC-Aircraft

### C.3.1 FGVC-Aircraft, test AUC at every `k`

| Method | 5 | 10 | 50 | 250 | 500 | 1,000 | 2,048 |
|---|---|---|---|---|---|---|---|
| Correlation (Spearman) | — | 0.736 | 0.766 | 0.813 | 0.828 | 0.828 | — |
| Mann-Whitney | — | 0.736 | 0.766 | 0.813 | 0.834 | 0.855 | — |
| Mann-Whitney 500, then RFE | — | 0.717 | 0.801 | 0.832 | — | — | — |
| Mann-Whitney 500, then tree importance | — | 0.743 | 0.781 | 0.814 | — | — | — |
| Mutual information | — | 0.714 | 0.751 | 0.802 | 0.827 | 0.850 | — |
| RFE | — | 0.715 | 0.792 | 0.842 | 0.855 | 0.862 | — |
| Random | — | 0.612 | 0.702 | 0.786 | 0.817 | 0.848 | 0.862 |
| Sequential | 0.700 | 0.703 | — | — | — | — | — |
| Tree importance | — | 0.739 | 0.778 | 0.806 | 0.826 | 0.851 | — |

### C.3.2 FGVC-Aircraft, at the chosen `k`

| Method | Test AUC [Q1, Q3] | Kept: median (min to max) | Selection, s | Search or validation, s | Stability |
|---|---|---|---|---|---|
| All features | 0.862 [0.850, 0.877] | 2,048 | 0.00 | — | — |
| RFE | 0.862 [0.850, 0.877] | 1,000 (500 to 1,000) | 0.31 | 5.63 | 0.27 |
| RFECV, by AUC | 0.858 [0.840, 0.870] | 1,028 (620 to 2,048) | 0.18 | 2.06 | 0.15 |
| Mann-Whitney | 0.854 [0.837, 0.870] | 1,000 (500 to 1,000) | 0.02 | 6.26 | 0.65 |
| RFECV, by accuracy | 0.854 [0.839, 0.868] | 926 (212 to 1,844) | 0.21 | 1.77 | 0.12 |
| Tree importance | 0.851 [0.840, 0.866] | 1,000 | 1.07 | 10.42 | 0.16 |
| Mutual information | 0.847 [0.834, 0.861] | 1,000 (500 to 1,000) | 0.14 | 6.01 | 0.27 |
| Lasso | 0.846 [0.831, 0.855] | 309 (132 to 336) | 0.16 | 3.25 | 0.31 |
| Mann-Whitney 500, then RFE | 0.829 [0.820, 0.841] | 250 (50 to 250) | 0.34 | 1.51 | 0.40 |
| Mann-Whitney 500, then Lasso | 0.828 [0.815, 0.840] | 248.5 (91 to 266) | 0.14 | 2.84 | 0.36 |
| Correlation (Spearman) | 0.824 [0.811, 0.838] | 419 (250 to 465) | 0.02 | 0.72 | 0.70 |
| HC2008 | 0.817 [0.811, 0.830] | 204 | 0.02 | — | 0.73 |
| Berk-Jones | 0.815 [0.811, 0.827] | 203 (183 to 204) | 0.02 | — | 0.73 |
| Mann-Whitney 500, then tree importance | 0.814 [0.807, 0.825] | 250 | 0.64 | 3.28 | 0.53 |
| HC2004 | 0.674 [0.660, 0.689] | 1 (1 to 2) | 0.01 | — | 0.60 |
