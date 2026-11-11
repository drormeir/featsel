"""
Copied images in BreastMNIST, found by geometric registration.

Pawłowska et al. (2023, Data in Brief 48, 109247) found that many images of
BUSI, the source of BreastMNIST, are copies of others: the same scan,
shifted, scaled or cropped, sometimes with another label. Their list names
the images by BUSI file name, which BreastMNIST does not keep. So this module
finds the copies in BreastMNIST's own images, by the letter's method: two
images are copies when a geometric transform maps one onto the other.

For each image, its 30 nearest images by ResNet-50 embedding are the
candidates. SIFT keypoints are matched between the two, keeping a match only
when it is clearly closer than the next best (Lowe's ratio, 0.75). A pair is
a copy when the matches fit one affine transform (RANSAC, within 2 pixels)
with at least 15 inliers, the transform overlaps the two images on at least
a fifth of the image, and the first image, warped by it, correlates with the
second at above 0.9 over that overlap. Copies join into groups through
shared images.

BreastMNIST runs this search once, on first use, and caches the groups
beside the downloaded images (BreastMNIST.copies). It needs OpenCV, in the
'embeddings' extra. demos/copies.ipynb shows the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

N_CANDIDATES = 30
MIN_INLIERS = 15
MIN_CORRELATION = 0.9


def copy_pairs(images: np.ndarray, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Every pair of images found to be copies, by image position.

    Columns: first, second, and the affine transform mapping the first onto
    the second, as its scale (the square root of its determinant) and its
    shift in pixels.
    """
    import cv2  # the 'embeddings' extra

    sift = cv2.SIFT.create()
    features = [sift.detectAndCompute(image, None) for image in images]

    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    candidates = {tuple(sorted((i, int(j)))) for i in range(len(images))
                  for j in np.argsort(-similarity[i])[:N_CANDIDATES]}

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for i, j in sorted(candidates):
        affine = copy_transform(cv2, matcher, images[i], images[j], features[i], features[j])
        if affine is not None:
            rows.append((i, j, np.sqrt(abs(np.linalg.det(affine[:, :2]))), np.hypot(*affine[:, 2])))
    return pd.DataFrame(rows, columns=['first', 'second', 'scale', 'shift'])


def copy_groups(pairs: pd.DataFrame, n_images: int) -> pd.Series:
    """
    Join copy pairs into groups through shared images.

    Returns the group of every image that has a copy, by image position.
    Groups are numbered by their first image, so the numbers do not depend
    on the order connected_components happens to use.
    """
    graph = coo_matrix((np.ones(len(pairs)), (pairs['first'], pairs['second'])),
                       shape=(n_images, n_images))
    _, component = connected_components(graph, directed=False)
    copied = np.flatnonzero(np.bincount(component)[component] > 1)
    labels = {c: g for g, c in enumerate(dict.fromkeys(component[copied]))}
    return pd.Series([labels[c] for c in component[copied]], index=copied, name='group')


def copy_transform(cv2, matcher, image_a, image_b, features_a, features_b) -> np.ndarray | None:
    """The affine transform mapping image_a onto image_b if they are copies, else None."""
    (keys_a, desc_a), (keys_b, desc_b) = features_a, features_b
    if desc_a is None or desc_b is None or len(keys_a) < 2 or len(keys_b) < 2:
        return None
    good = [m for m, second in (p for p in matcher.knnMatch(desc_a, desc_b, k=2) if len(p) == 2)
            if m.distance < 0.75 * second.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.array([keys_a[m.queryIdx].pt for m in good], dtype=np.float32)
    dst = np.array([keys_b[m.trainIdx].pt for m in good], dtype=np.float32)
    affine, inliers = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=2.0)
    if affine is None or inliers.sum() < MIN_INLIERS:
        return None
    size = image_b.shape[::-1]
    warped = cv2.warpAffine(image_a, affine, size)
    overlap = cv2.warpAffine(np.full_like(image_a, 255), affine, size) == 255
    if overlap.sum() < 0.2 * overlap.size:
        return None
    a, b = warped[overlap].astype(float), image_b[overlap].astype(float)
    if a.std() == 0 or b.std() == 0 or np.corrcoef(a, b)[0, 1] <= MIN_CORRELATION:
        return None
    return affine
