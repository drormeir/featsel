"""
This project's choices on FGVC-Aircraft: which images, which label, which features.

An experiment config names this module as its `dataset`. The runner reads
NAME and TARGET and calls `load_xy(target=...)`, with any `dataset_params`
of the config as further arguments.

The task: Boeing 737 against the Airbus A320 family (A318, A319, A320,
A321). Two families of narrow-body airliners that look alike; 800 against
400 images.
"""

from __future__ import annotations

import pandas as pd

from .aircraft import FGVCAircraft

NAME = 'FGVC-Aircraft ResNet-50'
TARGET = 'family'
FAMILIES = ('Boeing 737', 'A320')  # the archive's names; A320 holds A318 to A321
# Near-identical photos of the task, found by demos/duplicates.ipynb: one
# Boeing 737-900 shot seconds apart. All share one label.
NEAR_COPIES = (('0221294', '0223783', '0227126'),)


def load_xy(
    data: FGVCAircraft | None = None,
    target: str = TARGET,
    drop_copies: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return X (images x embedding dimensions) and y, the family.

    Keeps every image of the two families, from all three official splits:
    the runner draws its own splits. Drops embedding dimensions with a single
    value across the kept images. Pass `data` to reuse a loaded dataset.

    drop_copies keeps the first photo of each group in NEAR_COPIES and drops
    the others: 1,198 photos remain.
    """
    if data is None:
        data = FGVCAircraft()
    y = data.samples.loc[data.samples['family'].isin(FAMILIES), target]
    if drop_copies:
        y = y.drop([photo for group in NEAR_COPIES for photo in group[1:]])
    X = data.embeddings.loc[y.index]
    return X.loc[:, X.nunique() > 1], y
