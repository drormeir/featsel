"""
This project's choices on BreastMNIST: which images, which label, which features.

An experiment config names this module as its `dataset`. The runner reads
NAME and TARGET and calls `load_xy(target=...)`, with any `dataset_params`
of the config as further arguments.
"""

from __future__ import annotations

import pandas as pd

from .breastmnist import BreastMNIST

NAME = 'BreastMNIST ResNet-50'
TARGET = 'label'


def load_xy(
    data: BreastMNIST | None = None,
    target: str = TARGET,
    drop_copies: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return X (images x embedding dimensions) and y.

    Keeps all 780 images: MedMNIST's train/val/test split is ignored, because
    the runner draws its own splits. Drops embedding dimensions with a single
    value across the kept images. Pass `data` to reuse a loaded dataset.

    drop_copies removes the copied images (samples' copy_group). Of a group whose
    images share one label, the first image is kept. A group whose labels
    differ is removed whole, since its true label is unknown. 585 images
    remain: 180 malignant, 405 benign or normal.
    """
    if data is None:
        data = BreastMNIST()
    y = data.samples[target]
    if drop_copies:
        groups = data.samples['copy_group'].dropna()
        labels = y.loc[groups.index].groupby(groups)
        conflicting = groups.isin(labels.nunique().index[labels.nunique() > 1])
        repeated = groups.duplicated()
        y = y.drop(groups.index[conflicting | repeated])
    X = data.embeddings.loc[y.index]
    return X.loc[:, X.nunique() > 1], y
