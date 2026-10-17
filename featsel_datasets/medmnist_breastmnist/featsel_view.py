"""
This project's choices on BreastMNIST: which images, which label, which features.

An experiment config names this module as its `dataset`. The runner reads
NAME and TARGET and calls `load_xy(target=...)`.
"""

from __future__ import annotations

import pandas as pd

from .breastmnist import BreastMNIST

NAME = 'BreastMNIST ResNet-50'
TARGET = 'label'


def load_xy(
    data: BreastMNIST | None = None,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return X (images x embedding dimensions) and y.

    Keeps all 780 images: MedMNIST's train/val/test split is ignored, because
    the runner draws its own splits. Drops embedding dimensions with a single
    value across the images. Pass `data` to reuse a loaded dataset.
    """
    if data is None:
        data = BreastMNIST()
    X = data.embeddings
    return X.loc[:, X.nunique() > 1], data.samples[target]
