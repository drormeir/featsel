"""
Tests for the BreastMNIST loader on the real data.

They run only when the images and the embeddings are already in the cache,
because computing the embeddings needs PyTorch.
"""

import numpy as np
import pytest

from featsel_datasets.medmnist_breastmnist.breastmnist import FILE, BreastMNIST
from featsel_datasets.medmnist_breastmnist.copies import copy_groups
from featsel_datasets.medmnist_breastmnist.featsel_view import load_xy


@pytest.fixture(scope='module')
def data():
    data = BreastMNIST()
    if not all((data.cache_dir / name).exists() for name in (FILE, 'embeddings_resnet50.npz')):
        pytest.skip('BreastMNIST or its embeddings are not cached')
    return data


def test_images_and_labels_match_medmnist(data):
    """Test the published counts: 546/78/156 images, 210 malignant."""
    assert data.images.shape == (780, 224, 224)
    assert data.samples['split'].value_counts().to_dict() == {'train': 546, 'test': 156, 'val': 78}
    assert (data.samples['label'] == 'malignant').sum() == 210
    assert data.samples.index.is_unique


def test_load_xy_aligns_embeddings_with_labels(data):
    """Test that X holds one 2,048-dimension embedding per image, aligned with y."""
    X, y = load_xy(data)

    assert X.shape == (780, 2048)
    assert X.index.equals(y.index)
    assert X.index.equals(data.samples.index)
    assert (X.nunique() > 1).all()


def test_drop_copies_keeps_one_image_per_agreeing_group(data):
    """Test that one image of each copy group remains, none of a group with mixed labels."""
    groups = copy_groups()
    labels = data.samples['label']
    mixed = labels.loc[groups.index].groupby(groups).nunique() > 1

    X, y = load_xy(data, drop_copies=True)

    kept = groups.loc[groups.index.intersection(y.index)]
    assert X.index.equals(y.index)
    assert kept.is_unique
    assert set(kept) == set(mixed.index[~mixed])
    assert len(y) == 780 - (len(groups) - groups.nunique()) - mixed.sum() == 585


def test_copies_have_near_identical_embeddings(data):
    """Test that the first two images of a copy group embed almost alike, as copies should."""
    groups = copy_groups()
    emb = data.embeddings.loc[groups.index].to_numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    first = groups.groupby(groups).head(1).sort_values().index
    second = groups.groupby(groups).nth(1).sort_values().index

    within = (emb[groups.index.get_indexer(first)] * emb[groups.index.get_indexer(second)]).sum(1)

    assert np.median(within) > 0.95
