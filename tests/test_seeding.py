"""
Tests that every fit seeds the shared random generators first.
"""

import random

import numpy as np
import pytest

from featsel.selectors import FeatureSelector


@pytest.fixture
def shared_draw():
    """A selector that picks features from the shared generators, then is unregistered."""
    class SharedDraw(FeatureSelector):
        def __init__(self, n_features=3, random_state=42):
            self.n_features = n_features
            self.random_state = random_state

        def fit(self, X, y=None):
            self.n_features_in_ = X.shape[1]
            chosen = np.random.choice(X.shape[1], self.n_features, replace=False)
            self.draws_ = (random.random(), *chosen)
            self.support_ = self._mask(chosen)
            return self

    yield SharedDraw
    del FeatureSelector._registry['shareddraw']


X = np.zeros((5, 40))


def test_shared_draws_repeat(shared_draw):
    """Test that two fits draw the same, whatever was drawn in between."""
    first = shared_draw().fit(X).draws_
    np.random.random(100)
    random.random()

    assert shared_draw().fit(X).draws_ == first


def test_seed_follows_random_state(shared_draw):
    """Test that a different random_state draws differently."""
    assert shared_draw(random_state=1).fit(X).draws_ != shared_draw().fit(X).draws_


def test_none_leaves_shared_state_alone(shared_draw):
    """Test that random_state=None does not reseed: two fits draw differently."""
    assert shared_draw(random_state=None).fit(X).draws_ != shared_draw(random_state=None).fit(X).draws_


def test_torch_is_seeded(shared_draw):
    """Test that PyTorch's generator is seeded once PyTorch is imported."""
    torch = pytest.importorskip('torch')
    shared_draw().fit(X)
    first = torch.rand(3)
    shared_draw().fit(X)

    assert torch.equal(torch.rand(3), first)
