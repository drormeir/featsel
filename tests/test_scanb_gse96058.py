"""
Tests for the GSE96058 loader on the real data.

They run only when the files are already in the cache, because the first
download is 592 MB. Loading takes about a minute.
"""

import pytest

from featsel_datasets.scanb_gse96058.featsel_view import load_xy
from featsel_datasets.scanb_gse96058.gse96058 import EXPRESSION, GSE96058, MATRICES


@pytest.fixture(scope='module')
def data():
    data = GSE96058()
    if not all((data.cache_dir / name).exists() for name in (EXPRESSION, *MATRICES)):
        pytest.skip('GSE96058 is not downloaded')
    return data


def test_samples_and_genes_are_unique(data):
    """Test that sample IDs and gene names identify one entry each."""
    assert data.samples.index.is_unique
    assert data.expression.index.is_unique
    assert data.expression.columns.is_unique
    assert set(data.samples.index) == set(data.expression.index)


def test_load_xy_keeps_one_hiseq_sample_per_patient(data):
    """Test that X holds HiSeq samples only, one per patient, aligned with y."""
    X, y = load_xy(data)
    samples = data.samples.loc[X.index]

    assert X.index.equals(y.index)
    assert (samples['instrument model'] == 'HiSeq 2000').all()
    assert samples['specimen'].is_unique
    assert y.notna().all()
    assert (y == samples['pam50 subtype']).all()
    assert (X.nunique() > 1).all()
