# Bibliography

Works this project relies on. Where a copy exists, the last column gives its
file name. `local/` is ignored by git: the repo is public, and those files may
not be redistributed. Download missing papers from the link.

Local copies were downloaded on 2026-11-10 from legal sources only (arXiv,
journal open access, author and university pages). Each was checked to
open as a PDF and show the right title page. "Preprint" or "chapter" marks
a copy that is not the published version.

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
| Al-Dhabyani, W., Gomaa, M., Khaled, H. and Fahmy, A. (2020). Dataset of breast ultrasound images. *Data in Brief* 28, 104863. https://doi.org/10.1016/j.dib.2019.104863. The source images of BreastMNIST. | none: open access, but the hosts returned web pages, not the PDF |
| Pawłowska, A., Karwat, P. and Żołek, N. (2023). Letter to the Editor. Re: "Dataset of breast ultrasound images by W. Al-Dhabyani et al., Data in Brief, 2020, 28, 104863". *Data in Brief* 48, 109247. https://doi.org/10.1016/j.dib.2023.109247. 155 of the 780 BUSI images (19%) are copies, some labelled differently; 70 show the axilla; at least 7 a biopsy needle. Its supplement (Mendeley Data, doi 10.17632/k8t3gnx9h6.1) lists the pairs. | `local/Pawlowska_2023_BUSI_Letter.xml` (full text from Europe PMC) |
| Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., et al. (2024). Curated benchmark dataset for ultrasound based breast lesion analysis. *Scientific Data* 11, 148. https://doi.org/10.1038/s41597-024-02984-z. States that more than 40% of BUSI's images have significant defects. | `local/Pawlowska_2024_Curated_BUSI_Benchmark.pdf` |
| He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*. The ResNet-50 that embeds the BreastMNIST images. | `local/He_2016_ResNet.pdf` (arXiv) |

## Higher Criticism

| Work | Local copy |
|------|------------|
| Donoho, D. and Jin, J. (2004). Higher criticism for detecting sparse heterogeneous mixtures. *Annals of Statistics* 32(3), 962-994. https://arxiv.org/abs/math/0410072 | `local/Donoho_Jin_2004_Higher_Criticism.pdf` |
| Donoho, D. and Jin, J. (2008). Higher criticism thresholding: Optimal feature selection when useful features are rare and weak. *PNAS* 105(39), 14790-14795. https://pmc.ncbi.nlm.nih.gov/articles/PMC2553037/ | `local/Donoho_Jin_2008_HC_Thresholding_PNAS.pdf` (author's page) |
| Donoho, D. and Jin, J. (2009). Feature selection by higher criticism thresholding achieves the optimal phase diagram. *Philosophical Transactions of the Royal Society A* 367(1906), 4449-4470. https://doi.org/10.1098/rsta.2009.0129. Preprint 2008: https://arxiv.org/abs/0812.2263 | `local/Donoho_Jin_2008_HC_Thresholding_Phase_Diagram.pdf` |

## Evaluating selection methods

Cited in `final_report/03_how_to_measure.md`, sections 3.4 and 3.13.

| Work | Local copy |
|------|------------|
| Neekhra, B., Gupta, D. and Chakrabarti, P. P. (2025). On the (In)Significance of Feature Selection in High-Dimensional Datasets. https://arxiv.org/abs/2508.03593. Small random feature subsets matched or beat feature selection on 28 of 30 high-dimensional datasets, many from genomics. Bears directly on the random control. | `local/Neekhra_2025_Insignificance_Feature_Selection.pdf` (arXiv) |
| Ambroise, C. and McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS* 99(10), 6562-6566. | none: PNAS and PMC block automated download |
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
| Guyon, I. and Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research* 3, 1157-1182. https://www.jmlr.org/papers/v3/guyon03a.html | `local/Guyon_Elisseeff_2003_Variable_Feature_Selection.pdf` |
| Saeys, Y., Inza, I. and Larrañaga, P. (2007). A review of feature selection techniques in bioinformatics. *Bioinformatics* 23(19), 2507-2517. https://doi.org/10.1093/bioinformatics/btm344 | `local/Saeys_2007_Feature_Selection_Bioinformatics.pdf` |
| Kohavi, R. and John, G. H. (1997). Wrappers for feature subset selection. *Artificial Intelligence* 97(1-2), 273-324. https://doi.org/10.1016/S0004-3702(97)00043-X | none: the author's copy (ai.stanford.edu/~ronnyk/wrappersPrint.pdf) did not respond |
| Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B* 58(1), 267-288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x | `local/Tibshirani_1996_Lasso.pdf` |
| Breiman, L. (2001). Random forests. *Machine Learning* 45(1), 5-32. https://doi.org/10.1023/A:1010933404324 | `local/Breiman_2001_Random_Forests.pdf` |
| Guyon, I., Weston, J., Barnhill, S. and Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning* 46(1-3), 389-422. https://doi.org/10.1023/A:1012487302797. SVM-RFE. | `local/Guyon_2002_SVM_RFE.pdf` |
| Mann, H. B. and Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics* 18(1), 50-60. https://doi.org/10.1214/aoms/1177730491 | `local/Mann_Whitney_1947_U_Test.pdf` |
| Hanley, J. A. and McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29-36. https://doi.org/10.1148/radiology.143.1.7063747. The U statistic is the AUC. | `local/Hanley_McNeil_1982_ROC_Area.pdf` (author's scan) |
| Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B* 57(1), 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x | none: no legal free copy found |
| Dudoit, S., Fridlyand, J. and Speed, T. P. (2002). Comparison of discrimination methods for the classification of tumors using gene expression data. *Journal of the American Statistical Association* 97(457), 77-87. https://doi.org/10.1198/016214502753479248 | `local/Dudoit_2002_Discrimination_Methods.pdf` |
| Statnikov, A., Aliferis, C. F., Tsamardinos, I., et al. (2005). A comprehensive evaluation of multicategory classification methods for microarray gene expression cancer diagnosis. *Bioinformatics* 21(5), 631-643. https://doi.org/10.1093/bioinformatics/bti033 | none: no legal free copy found |
| Bolón-Canedo, V., Sánchez-Maroño, N., Alonso-Betanzos, A., et al. (2014). A review of microarray datasets and applied feature selection methods. *Information Sciences* 282, 111-135. https://doi.org/10.1016/j.ins.2014.05.042 | `local/Bolon_Canedo_2014_Microarray_Review.pdf` |
| Bengio, Y. and Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *Journal of Machine Learning Research* 5, 1089-1105. https://www.jmlr.org/papers/v5/grandvalet04a.html | `local/Bengio_Grandvalet_2004_KFold_Variance.pdf` |
| Kuncheva, L. I. (2007). A stability index for feature selection. *Proceedings of the 25th IASTED International Multi-Conference: Artificial Intelligence and Applications*, Innsbruck, 390-395. The stability index. | none: only on ResearchGate |

## Added with the final report

Cited in `final_report/`. Beyer, Friedman and Ledoit-Wolf checked on
Crossref, 2026-11-10; the rest against their local copies where one exists.
Bellman (1957) is not checked.

| Work | Local copy |
|------|------------|
| Bellman, R. (1957). *Dynamic Programming*. Princeton University Press. The curse of dimensionality. | none: a book |
| Beyer, K., Goldstein, J., Ramakrishnan, R. and Shaft, U. (1999). When is "nearest neighbor" meaningful? *Database Theory - ICDT'99*, LNCS, 217-235. https://doi.org/10.1007/3-540-49257-7_15 | `local/Beyer_1999_Nearest_Neighbor_Meaningful.pdf` |
| Friedman, J. H. (1989). Regularized discriminant analysis. *Journal of the American Statistical Association* 84(405), 165-175. https://doi.org/10.1080/01621459.1989.10478752 | `local/Friedman_1989_Regularized_Discriminant_Analysis.pdf` (preprint, SLAC-PUB-4389, 1988) |
| Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI*. Leave-one-out has high variance. | `local/Kohavi_1995_Cross_Validation_Bootstrap.pdf` (IJCAI proceedings) |
| Kubat, M. and Matwin, S. (1997). Addressing the curse of imbalanced training sets: one-sided selection. *ICML*. G-mean. | `local/Kubat_Matwin_1997_One_Sided_Selection.pdf` |
| Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2), 365-411. https://doi.org/10.1016/S0047-259X(03)00096-4. Shrinkage LDA. | `local/Ledoit_Wolf_2004_Covariance_Shrinkage.pdf` (preprint) |
| Chapelle, O. (2007). Training a support vector machine in the primal. *Neural Computation* 19(5), 1155-1178. https://doi.org/10.1162/neco.2007.19.5.1155. RFE's Newton solver. | `local/Chapelle_2007_SVM_Primal_chapter.pdf` (chapter version, *Large-Scale Kernel Machines*, MIT Press) |
| Liu, Y.-J. and Zhou, W. (2023). Dual Newton proximal point algorithm for solution paths of the ℓ1-regularized logistic regression. https://arxiv.org/abs/2310.19353. Lasso's GPU solver (PPDNA). | `local/Liu_Zhou_2023_PPDNA.pdf` (arXiv) |
| Meinshausen, N. and Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society, Series B* 72(4), 417-473. https://doi.org/10.1111/j.1467-9868.2010.00740.x | `local/Meinshausen_Buhlmann_2010_Stability_Selection.pdf` (arXiv preprint) |
| Ross, B. C. (2014). Mutual information between discrete and continuous data sets. *PLoS ONE* 9(2), e87357. https://doi.org/10.1371/journal.pone.0087357. scikit-learn's mutual information estimate. | `local/Ross_2014_Mutual_Information.pdf` |
| Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin* 1(6), 80-83. https://doi.org/10.2307/3001968. The paired tests of chapter 7. | `local/Wilcoxon_1945_Ranking_Methods.pdf` |

## Notes on Higher Criticism

The 2004 paper defines the Higher Criticism statistic, for detection. The
phase-diagram paper defines Higher Criticism thresholding, for feature
selection, in Definitions 1.2 and 1.3.

Both papers centre the i-th sorted p-value at i/N (2004, section 1.2;
phase-diagram paper, Definition 1.2). The project's HC2008 uses i/(N+1),
the exact mean, instead: a small change that is not in the papers.
