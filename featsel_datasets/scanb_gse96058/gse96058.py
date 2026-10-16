"""
GSE96058: SCAN-B breast cancer RNA-seq, 3,273 tumours (Brueffer et al., 2018).

Downloads the published files once, keeps them next to this file as a cache,
and exposes them unchanged: every sample column and every sample field, under
GSE96058's own names. It makes no choice about which samples, genes or labels
to use. That is left to the caller.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058'
EXPRESSION = 'GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz'
MATRICES = (
    'GSE96058-GPL11154_series_matrix.txt.gz',  # HiSeq 2000
    'GSE96058-GPL18573_series_matrix.txt.gz',  # NextSeq 500
)
URLS = {EXPRESSION: f'{_BASE}/suppl/{EXPRESSION}'} | {m: f'{_BASE}/matrix/{m}' for m in MATRICES}


class GSE96058:
    """
    The GSE96058 dataset, downloaded on first use.

    Usage:
        data = GSE96058()
        data.expression   # 3,409 sample columns x 30,865 genes
        data.samples      # one row per sample column: labels and clinical fields

    Both are indexed by the sample title (F1, F2, ...). A tumour sequenced
    twice has two rows; `samples['specimen']` identifies the tumour.
    """

    def __init__(self, cache_dir: str | Path = Path(__file__).parent):
        self.cache_dir = Path(cache_dir)

    def _path(self, name: str) -> Path:
        """Return the cached file, downloading it first if it is missing."""
        path = self.cache_dir / name
        if not path.exists():
            part = path.with_name(path.name + '.part')
            print(f'Downloading {URLS[name]}')
            with urllib.request.urlopen(URLS[name]) as response, open(part, 'wb') as f:
                shutil.copyfileobj(response, f)
            part.rename(path)
        return path

    @cached_property
    def expression(self) -> pd.DataFrame:
        """Gene expression, one row per sample column, one column per gene."""
        return pd.read_csv(self._path(EXPRESSION), index_col=0).T

    @cached_property
    def samples(self) -> pd.DataFrame:
        """Every sample field from both series matrices, 'NA' read as missing."""
        samples = pd.concat(_read_matrix(self._path(m)) for m in MATRICES).replace('NA', np.nan)
        for column in samples:
            try:
                samples[column] = pd.to_numeric(samples[column])
            except ValueError:
                pass
        # The external id reads Q<case>.C<case>.S<specimen>....; repeats share the specimen.
        samples['specimen'] = samples['scan-b external id'].str.split('.').str[2]
        return samples


def _read_matrix(path: Path) -> pd.DataFrame:
    """Read the sample title, accession, platform and characteristics of one series matrix."""
    fields = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            if not line.startswith('!Sample_'):
                continue
            key, *values = (v.strip('"') for v in line.rstrip('\n').split('\t'))
            if key == '!Sample_characteristics_ch1':
                fields[values[0].split(': ', 1)[0]] = [v.split(': ', 1)[1] for v in values]
            elif key in ('!Sample_title', '!Sample_geo_accession', '!Sample_platform_id'):
                fields[key.removeprefix('!Sample_')] = values
    return pd.DataFrame(fields).set_index('title')
