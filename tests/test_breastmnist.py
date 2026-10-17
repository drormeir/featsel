"""
Tests for the BreastMNIST loader on the real data.

They run only when the images and the embeddings are already in the cache,
because computing the embeddings needs PyTorch.
"""

import pytest

from featsel_datasets.medmnist_breastmnist.breastmnist import FILE, BreastMNIST
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
