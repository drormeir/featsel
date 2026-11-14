"""
This project's choices on GSE96058: which samples, which label, which genes.

An experiment config names this module as its `dataset`. The runner reads
NAME and TARGET and calls `load_xy(target=...)`.
"""

from __future__ import annotations

import pandas as pd

from .gse96058 import GSE96058

NAME = 'SCAN-B GSE96058'
TARGET = 'pam50 subtype'


def load_xy(
    data: GSE96058 | None = None,
    target: str = TARGET,
    instrument: str = 'HiSeq 2000',
    one_per_specimen: bool = True,
    min_spread: float = 1e-2,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return X (samples x genes) and y for one target.

    Keeps the samples sequenced on `instrument`, one sample per patient if
    `one_per_specimen` (the first in file order), and the samples whose
    target is known. Drops flat genes: those whose range across those
    samples, divided by their largest absolute value, is below `min_spread`.
    On SCAN-B these 11 genes sit at one value in almost every patient, and
    standardizing them in float32 turns them into noise.
    Pass `data` to reuse a loaded dataset; otherwise it is read from the cache.
    """
    if data is None:
        data = GSE96058()
    samples = data.samples[data.samples['instrument model'] == instrument]
    if one_per_specimen:
        samples = samples[~samples['specimen'].duplicated()]
    y = samples[target].dropna()
    X = data.expression.loc[y.index]
    spread = (X.max() - X.min()) / X.abs().max()
    return X.loc[:, spread >= min_spread], y
