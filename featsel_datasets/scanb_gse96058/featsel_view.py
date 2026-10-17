"""This project's choices on GSE96058: which samples, which label, which genes."""

from __future__ import annotations

import pandas as pd

from .gse96058 import GSE96058


def load_xy(
    data: GSE96058 | None = None,
    target: str = 'pam50 subtype',
    instrument: str = 'HiSeq 2000',
    one_per_specimen: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return X (samples x genes) and y for one target.

    Keeps the samples sequenced on `instrument`, one sample per patient if
    `one_per_specimen` (the first in file order), and the samples whose
    target is known. Drops genes with a single value across those samples.
    Pass `data` to reuse a loaded dataset; otherwise it is read from the cache.
    """
    if data is None:
        data = GSE96058()
    samples = data.samples[data.samples['instrument model'] == instrument]
    if one_per_specimen:
        samples = samples[~samples['specimen'].duplicated()]
    y = samples[target].dropna()
    X = data.expression.loc[y.index]
    return X.loc[:, X.nunique() > 1], y
