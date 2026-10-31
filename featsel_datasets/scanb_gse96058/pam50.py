"""
The 50 genes of the PAM50 subtype classifier (Parker et al., J Clin Oncol 2009).

Source: the rows of pam50_centroids.txt, published by the Perou lab at
https://genome-publications.bioinf.unc.edu/PAM50/PAM50_R/pam50_centroids.txt
(read 2026-10-31).

Three symbols there are old names. GSE96058 uses the current ones, so the
list below does too: CDCA1 is NUF2, KNTC2 is NDC80, ORC6L is ORC6.
"""

PAM50_GENES = [
    'ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1', 'CDC20', 'CDC6',
    'NUF2', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2', 'ESR1', 'EXO1', 'FGFR4',
    'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C', 'NDC80', 'KRT14', 'KRT17', 'KRT5', 'MAPT',
    'MDM2', 'MELK', 'MIA', 'MKI67', 'MLPH', 'MMP11', 'MYBL2', 'MYC', 'NAT1', 'ORC6',
    'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B', 'TYMS', 'UBE2C', 'UBE2T',
]
