# Bibliography

Works this project relies on. Where a copy exists, the last column gives its
file name. `local/` is ignored by git: the repo is public, and those files may
not be redistributed. Download missing papers from the link.

## Project

| Work | Local copy |
|------|------------|
| Reichman University, MLDS program. Feature selection final project: specification. | `MLDS_Feature_Selection.pdf` (in git) |
| An earlier feature selection report on SCAN-B. Unpublished, about 3,900 words. | `local/Final Project Report with analysis.docx` |

## Dataset

| Work | Local copy |
|------|------------|
| Brueffer, C., Vallon-Christersson, J., Grabau, D., et al. (2018). Clinical value of RNA sequencing-based classifiers for prediction of the five conventional breast cancer biomarkers: a report from the population-based multicenter Sweden Cancerome Analysis Network-Breast Initiative. *JCO Precision Oncology* 2, 1-18. https://doi.org/10.1200/PO.17.00135. Data: GEO GSE96058. | `local/Brueffer_2018_SCANB_Biomarker_Classifiers.pdf` |
| Data Supplement to the above: methods PDF and Supplementary Tables 1-9. Table 3 holds the multigene classifiers' shrunken centroids per gene; Table 6 the single-gene cutoffs. https://ascopubs.org/doi/suppl/10.1200/PO.17.00135 | `local/Brueffer_2018_supplement/` |
| Parker, J. S., Mullins, M., Cheang, M. C. U., et al. (2009). Supervised risk predictor of breast cancer based on intrinsic subtypes. *Journal of Clinical Oncology* 27(8), 1160-1167. https://doi.org/10.1200/JCO.2008.18.1370. Defines PAM50. Supplemental Table 2 lists the 50 genes plus 5 control genes (ACTB, MRPL19, PSMC4, RPLP0, SF3A1); `pam50_genes.csv` maps the 50 to current gene symbols. | `local/Parker_2009_PAM50/` |
| Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H. and Ni, B. (2023). MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*. https://arxiv.org/abs/2110.14795. BreastMNIST, the second dataset; 224 x 224 files on Zenodo, record 10519652. | `local/Yang_2023_MedMNIST_v2.pdf` (arXiv v2) |
| Al-Dhabyani, W., Gomaa, M., Khaled, H. and Fahmy, A. (2020). Dataset of breast ultrasound images. *Data in Brief* 28, 104863. https://doi.org/10.1016/j.dib.2019.104863. The source images of BreastMNIST. | none |
| He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*. The ResNet-50 that embeds the BreastMNIST images. | none |

## Higher Criticism

| Work | Local copy |
|------|------------|
| Donoho, D. and Jin, J. (2004). Higher criticism for detecting sparse heterogeneous mixtures. *Annals of Statistics* 32(3), 962-994. https://arxiv.org/abs/math/0410072 | `local/Donoho_Jin_2004_Higher_Criticism.pdf` |
| Donoho, D. and Jin, J. (2008). Higher criticism thresholding: Optimal feature selection when useful features are rare and weak. *PNAS* 105(39), 14790-14795. https://pmc.ncbi.nlm.nih.gov/articles/PMC2553037/ | none |
| Donoho, D. and Jin, J. (2009). Feature selection by higher criticism thresholding achieves the optimal phase diagram. *Philosophical Transactions of the Royal Society A* 367(1906), 4449-4470. https://doi.org/10.1098/rsta.2009.0129. Preprint 2008: https://arxiv.org/abs/0812.2263 | `local/Donoho_Jin_2008_HC_Thresholding_Phase_Diagram.pdf` |

## Evaluating selection methods

Cited in `docs/selection_strength.md`.

| Work | Local copy |
|------|------------|
| Neekhra, B., Gupta, D. and Chakrabarti, P. P. (2025). On the (In)Significance of Feature Selection in High-Dimensional Datasets. https://arxiv.org/abs/2508.03593. Small random feature subsets matched or beat feature selection on 28 of 30 high-dimensional datasets, many from genomics. Bears directly on the random control. | none |
| Ambroise, C. and McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS* 99(10), 6562-6566. | none |
| Rajabinasab, M., Lautrup, A. D., Hyrup, T. and Zimek, A. (2024). FSDEM: Feature Selection Dynamic Evaluation Metric. SISAP 2024. https://arxiv.org/abs/2408.14234 | `local/Rajabinasab_2024_FSDEM.pdf` |
| Overschie, J. G. S. (2021). A novel evaluation methodology for supervised Feature Ranking algorithms. Technical report, University of Groningen. https://arxiv.org/abs/2207.04258 | `local/Overschie_2021_Feature_Ranking_Evaluation.pdf` |
| Slavkov, I., Petković, M., Geurts, P., Kocev, D. and Džeroski, S. (2020). Error curves for evaluating the quality of feature rankings. *PeerJ Computer Science* 6, e310. https://doi.org/10.7717/peerj-cs.310 | none: the site blocks automated download |
| Glaysher, P., Katzy, J. M. and An, S. (2019). Iterative subtraction method for Feature Ranking. https://arxiv.org/abs/1906.05718 | `local/Glaysher_2019_Iterative_Subtraction.pdf` |
| Tian, C. J., Lv, J. and Xu, X. F. (2021). Evaluation of Feature Selection Methods for Mammographic Breast Cancer Diagnosis in a Unified Framework. *BioMed Research International* 2021, 6079163. https://doi.org/10.1155/2021/6079163 | none: the site blocks automated download |

## Methods and comparative studies

Cited in the report, chapters 2, 4 and 7. Checked on Crossref or jmlr.org,
2026-10-31; Kuncheva (2007) against citing papers.

| Work | Local copy |
|------|------------|
| Guyon, I. and Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research* 3, 1157-1182. https://www.jmlr.org/papers/v3/guyon03a.html | none |
| Saeys, Y., Inza, I. and Larrañaga, P. (2007). A review of feature selection techniques in bioinformatics. *Bioinformatics* 23(19), 2507-2517. https://doi.org/10.1093/bioinformatics/btm344 | none |
| Kohavi, R. and John, G. H. (1997). Wrappers for feature subset selection. *Artificial Intelligence* 97(1-2), 273-324. https://doi.org/10.1016/S0004-3702(97)00043-X | none |
| Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B* 58(1), 267-288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x | none |
| Breiman, L. (2001). Random forests. *Machine Learning* 45(1), 5-32. https://doi.org/10.1023/A:1010933404324 | none |
| Guyon, I., Weston, J., Barnhill, S. and Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning* 46(1-3), 389-422. https://doi.org/10.1023/A:1012487302797. SVM-RFE. | none |
| Mann, H. B. and Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics* 18(1), 50-60. https://doi.org/10.1214/aoms/1177730491 | none |
| Hanley, J. A. and McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29-36. https://doi.org/10.1148/radiology.143.1.7063747. The U statistic is the AUC. | none |
| Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B* 57(1), 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x | none |
| Dudoit, S., Fridlyand, J. and Speed, T. P. (2002). Comparison of discrimination methods for the classification of tumors using gene expression data. *Journal of the American Statistical Association* 97(457), 77-87. https://doi.org/10.1198/016214502753479248 | none |
| Statnikov, A., Aliferis, C. F., Tsamardinos, I., et al. (2005). A comprehensive evaluation of multicategory classification methods for microarray gene expression cancer diagnosis. *Bioinformatics* 21(5), 631-643. https://doi.org/10.1093/bioinformatics/bti033 | none |
| Bolón-Canedo, V., Sánchez-Maroño, N., Alonso-Betanzos, A., et al. (2014). A review of microarray datasets and applied feature selection methods. *Information Sciences* 282, 111-135. https://doi.org/10.1016/j.ins.2014.05.042 | none |
| Bengio, Y. and Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *Journal of Machine Learning Research* 5, 1089-1105. https://www.jmlr.org/papers/v5/grandvalet04a.html | none |
| Kuncheva, L. I. (2007). A stability index for feature selection. *Proceedings of the 25th IASTED International Multi-Conference: Artificial Intelligence and Applications*, Innsbruck, 390-395. The stability index. | none |

## Notes on Higher Criticism

The 2004 paper defines the Higher Criticism statistic, for detection. The
phase-diagram paper defines Higher Criticism thresholding, for feature
selection, in Definitions 1.2 and 1.3.
