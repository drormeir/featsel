"""
BreastMNIST: 780 breast ultrasound images, malignant vs benign or normal.

Part of MedMNIST v2 (Yang et al., 2023), from the images of Al-Dhabyani et al.
(2020). This module downloads the 224 x 224 version once, keeps it next to
this file as a cache, and exposes it unchanged. It also turns each image into
a 2,048-number ResNet-50 embedding, computed once and cached.

PyTorch is needed only to compute the embeddings. Reading the images, or
embeddings already cached, does not import it.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

FILE = 'breastmnist_224.npz'
URL = f'https://zenodo.org/records/10519652/files/{FILE}?download=1'
MD5 = 'b56378a6eefa9fed602bb16d192d4c8b'  # published on the Zenodo record
SPLITS = ('train', 'val', 'test')
LABELS = {0: 'malignant', 1: 'benign_or_normal'}  # MedMNIST v2 merges benign and normal


class BreastMNIST:
    """
    The BreastMNIST dataset at 224 x 224, downloaded on first use.

    Usage:
        data = BreastMNIST()
        data.images       # (780, 224, 224) uint8, in the order of data.samples
        data.samples      # one row per image: 'label' and MedMNIST's 'split'
        data.embeddings   # 780 images x 2,048 ResNet-50 features

    Images are indexed train_000, ..., val_000, ..., test_000, ...
    """

    def __init__(self, cache_dir: str | Path = Path(__file__).parent):
        self.cache_dir = Path(cache_dir)

    def _file(self) -> Path:
        """Return the cached file, downloading and checking it first if missing."""
        path = self.cache_dir / FILE
        if not path.exists():
            part = path.with_name(path.name + '.part')
            print(f'Downloading {URL}')
            with urllib.request.urlopen(URL) as response, open(part, 'wb') as f:
                shutil.copyfileobj(response, f)
            if hashlib.md5(part.read_bytes()).hexdigest() != MD5:
                raise ValueError(f'{part} does not match the published MD5')
            part.rename(path)
        return path

    @cached_property
    def _arrays(self) -> dict[str, np.ndarray]:
        with np.load(self._file()) as npz:
            return {name: npz[name] for name in npz.files}

    @cached_property
    def images(self) -> np.ndarray:
        """All images, train then val then test, as uint8 grayscale."""
        return np.concatenate([self._arrays[f'{s}_images'] for s in SPLITS])

    @cached_property
    def samples(self) -> pd.DataFrame:
        """One row per image: its label and its MedMNIST split."""
        parts = []
        for split in SPLITS:
            labels = self._arrays[f'{split}_labels'][:, 0]
            index = [f'{split}_{i:03d}' for i in range(len(labels))]
            parts.append(pd.DataFrame(
                {'label': [LABELS[v] for v in labels], 'split': split}, index=index))
        return pd.concat(parts)

    @cached_property
    def embeddings(self) -> pd.DataFrame:
        """ResNet-50 embeddings, one row per image; computed once, then cached."""
        path = self.cache_dir / 'embeddings_resnet50.npz'
        if not path.exists():
            np.savez_compressed(path, embeddings=_resnet50_embeddings(self.images))
        with np.load(path) as npz:
            values = npz['embeddings']
        columns = [f'resnet50_{i:04d}' for i in range(values.shape[1])]
        return pd.DataFrame(values, index=self.samples.index, columns=columns)


def _resnet50_embeddings(images: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """
    Embed grayscale images with ImageNet ResNet-50, without its last layer.

    Each image is repeated over three channels and normalized as ImageNet
    images are. The output is the 2,048-number average-pooled feature vector.
    Uses the GPU when one is available.
    """
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size]).to(device)
            batch = batch.float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
            out.append(model((batch - mean) / std).cpu().numpy())
    return np.concatenate(out)
