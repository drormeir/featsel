"""
FGVC-Aircraft 2013b: 10,000 photos of aircraft, labelled by model variant.

Maji et al. (2013), arXiv:1306.5151. 100 variants with 100 images each,
grouped into families and manufacturers. This module downloads the archive
once (2.75 GB), keeps it next to this file as a cache, and exposes every
image's labels unchanged. It also turns each image into a 2,048-number
ResNet-50 embedding, computed once and cached, with a 64-bit difference
hash of each image for finding near-duplicates.

The archive is read in one pass and never unpacked. The images are for
non-commercial research only and are not redistributed.

PyTorch is needed only to compute the embeddings. Reading the labels, or
embeddings already cached, does not import it.
"""

from __future__ import annotations

import io
import shutil
import tarfile
import urllib.request
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

FILE = 'fgvc-aircraft-2013b.tar.gz'
URL = f'https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/{FILE}'
SIZE = 2_753_340_328  # bytes; the site publishes no checksum
ROOT = 'fgvc-aircraft-2013b/data'
SPLITS = ('train', 'val', 'test')
LEVELS = ('variant', 'family', 'manufacturer')
BANNER = 20  # pixels of copyright text at the bottom of every image


class FGVCAircraft:
    """
    The FGVC-Aircraft 2013b dataset, downloaded on first use.

    Usage:
        data = FGVCAircraft()
        data.samples      # one row per image: variant, family, manufacturer, split
        data.embeddings   # 10,000 images x 2,048 ResNet-50 features
        data.hashes       # 64-bit difference hash per image, for near-duplicates
        data.images(ids)  # some images, banner cropped, as PIL images

    Images are indexed by the archive's own image id, such as '1025794'.
    """

    def __init__(self, cache_dir: str | Path = Path(__file__).parent):
        self.cache_dir = Path(cache_dir)

    def _file(self) -> Path:
        """Return the cached archive, downloading it first if missing."""
        path = self.cache_dir / FILE
        if not path.exists():
            part = path.with_name(path.name + '.part')
            print(f'Downloading {URL} (2.75 GB)')
            with urllib.request.urlopen(URL) as response, open(part, 'wb') as f:
                shutil.copyfileobj(response, f)
            if part.stat().st_size != SIZE:
                raise ValueError(f'{part} has {part.stat().st_size} bytes, expected {SIZE}')
            part.rename(path)
        return path

    @cached_property
    def samples(self) -> pd.DataFrame:
        """One row per image: its variant, family, manufacturer and official split."""
        path = self.cache_dir / 'samples.csv.gz'
        if not path.exists():
            self._read_labels().to_csv(path)
        return pd.read_csv(path, index_col='image', dtype={'image': str})

    def _read_labels(self) -> pd.DataFrame:
        """Read the label files from the archive: one 'id label' line per image."""
        wanted = {f'{ROOT}/images_{level}_{split}.txt': (level, split)
                  for level in LEVELS for split in SPLITS}
        found: dict[tuple[str, str], dict[str, str]] = {}
        with tarfile.open(self._file(), 'r:gz') as tar:
            for member in tar:
                if member.name in wanted:
                    text = _read(tar, member).decode()
                    found[wanted[member.name]] = dict(
                        line.split(' ', 1) for line in text.splitlines() if line)
        rows = []
        for split in SPLITS:
            for image in found[('variant', split)]:
                rows.append({'image': image, 'split': split,
                             **{level: found[(level, split)][image] for level in LEVELS}})
        return pd.DataFrame(rows).set_index('image')

    @cached_property
    def _embedded(self) -> dict[str, np.ndarray]:
        """Embeddings and hashes, computed in one pass over the archive, then cached."""
        path = self.cache_dir / 'embeddings_resnet50_v1.npz'
        if not path.exists():
            ids, embeddings, hashes = _embed_archive(self._file())
            np.savez_compressed(path, ids=ids, embeddings=embeddings, hashes=hashes)
        with np.load(path) as npz:
            return {name: npz[name] for name in npz.files}

    @cached_property
    def embeddings(self) -> pd.DataFrame:
        """ResNet-50 embeddings, one row per image, in the order of samples."""
        values = self._embedded['embeddings']
        columns = [f'resnet50_{i:04d}' for i in range(values.shape[1])]
        frame = pd.DataFrame(values, index=self._embedded['ids'], columns=columns)
        return frame.loc[self.samples.index]

    @cached_property
    def hashes(self) -> pd.Series:
        """64-bit difference hash of each banner-cropped image, in the order of samples."""
        series = pd.Series(self._embedded['hashes'], index=self._embedded['ids'])
        return series.loc[self.samples.index]

    def images(self, ids) -> dict:
        """Some images, banner cropped, by id. One pass over the archive: about a minute."""
        wanted = {f'{ROOT}/images/{image}.jpg': image for image in ids}
        found = {}
        with tarfile.open(self._file(), 'r:gz') as tar:
            for member in tar:
                if member.name in wanted:
                    found[wanted[member.name]] = _open_cropped(_read(tar, member))
                    if len(found) == len(wanted):
                        break
        missing = set(wanted.values()) - set(found)
        if missing:
            raise KeyError(sorted(missing))
        return {image: found[image] for image in ids}


def _read(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    """The bytes of a regular file in the archive."""
    f = tar.extractfile(member)
    if f is None:
        raise ValueError(f'{member.name} is not a regular file')
    return f.read()


def _open_cropped(data: bytes):
    """Decode a JPEG and cut off the copyright banner at its bottom."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert('RGB')
    return img.crop((0, 0, img.width, img.height - BANNER))


def _dhash(img, size: int = 8) -> np.uint64:
    """Difference hash: the sign of each horizontal step on a 9 x 8 grayscale thumbnail."""
    small = np.asarray(img.convert('L').resize((size + 1, size)), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return np.uint64(int(''.join('1' if b else '0' for b in bits), 2))


def _embed_archive(path: Path, batch_size: int = 32):
    """
    Embed every image in the archive with ImageNet ResNet-50, without its last layer.

    Each image is banner-cropped, then given the weights' own preprocessing:
    resize to 256, centre crop to 224, ImageNet normalization. The output is
    the 2,048-number average-pooled feature vector. Uses the GPU when one is
    available, in 32-bit floats (16-bit is slower on a GTX 1650).

    The weights are torchvision's IMAGENET1K_V1, not the V2 that BreastMNIST
    uses. On the 737 vs A320 task, V2's features were 80% exact zeros and
    gave AUC 0.749 with all features; V1's were 0.1% zeros and gave 0.863.
    """
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    ids: list[str] = []
    hashes: list[np.uint64] = []
    out: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush():
        with torch.no_grad():
            out.append(model(torch.stack(batch).to(device)).cpu().numpy())
        batch.clear()

    with tarfile.open(path, 'r:gz') as tar:
        for member in tar:
            name = member.name
            if not (name.startswith(f'{ROOT}/images/') and name.endswith('.jpg')):
                continue
            img = _open_cropped(_read(tar, member))
            ids.append(Path(name).stem)
            hashes.append(_dhash(img))
            batch.append(preprocess(img))
            if len(batch) == batch_size:
                flush()
    if batch:
        flush()
    return np.array(ids), np.concatenate(out), np.array(hashes, dtype=np.uint64)
