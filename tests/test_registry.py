"""
Tests for selector self-registration and the BaseSelector.create factory.
"""

import numpy as np
import pytest

from featsel.selectors import ANOVAFSelector, BaseSelector

BUILT_IN = ['VarianceThreshold', 'ANOVAFSelector', 'MutualInfoSelector',
            'CorrelationSelector', 'RandomSelector', 'LassoSelector',
            'TreeImportanceSelector']


@pytest.fixture(autouse=True)
def restore_registry():
    """Classes defined inside a test must not leak into the next one."""
    saved = dict(BaseSelector._registry)
    yield
    BaseSelector._registry.clear()
    BaseSelector._registry.update(saved)


class _Stub(BaseSelector):
    """Concrete enough to register; the tests only check names."""

    def fit(self, X, y=None):
        return self

    def get_support(self, indices=False):
        return np.array([])


del BaseSelector._registry['_stub']  # a test helper, not a selector


def test_built_in_selectors_register_by_class_name():
    """Test that every built-in selector is reachable by its class name."""
    for name in BUILT_IN:
        assert BaseSelector._registry[name.lower()].__name__ == name


def test_create_is_case_insensitive_and_passes_params():
    """Test that any casing finds the class and parameters reach its constructor."""
    selector = BaseSelector.create('anovafselector', n_features=5)

    assert isinstance(selector, ANOVAFSelector)
    assert selector.n_features == 5


def test_unknown_name_lists_what_is_available():
    """Test that a typo in the name fails with the valid names."""
    with pytest.raises(ValueError, match="Unknown selector 'anova'.*anovafselector"):
        BaseSelector.create('anova')


def test_aliases_accept_a_string_or_a_list():
    """Test that aliases register alongside the class name."""
    class OneAlias(_Stub, aliases='one'):
        pass

    class TwoAliases(_Stub, aliases=['two', 'Deux']):
        pass

    assert BaseSelector._registry['one'] is OneAlias
    assert BaseSelector._registry['onealias'] is OneAlias
    assert BaseSelector._registry['two'] is TwoAliases
    assert isinstance(BaseSelector.create('DEUX'), TwoAliases)


def test_alias_equal_to_the_class_name_is_ignored():
    """Test that repeating the class name as an alias is not a collision."""
    class Echo(_Stub, aliases='ECHO'):
        pass

    assert BaseSelector._registry['echo'] is Echo


def test_name_collision_is_rejected_at_definition():
    """Test that two classes cannot share a name, whatever the casing."""
    class First(_Stub, aliases='shared'):
        pass

    with pytest.raises(ValueError, match="'shared' of Second is already taken by First"):
        class Second(_Stub, aliases='SHARED'):
            pass


def test_aliases_are_not_inherited():
    """Test that a subclass does not re-register its parent's aliases."""
    class Parent(_Stub, aliases='family'):
        pass

    class Child(Parent):
        pass

    assert BaseSelector._registry['family'] is Parent
    assert BaseSelector._registry['child'] is Child


def test_abstract_classes_do_not_register():
    """Test that a class leaving an abstract method open stays out."""
    class HalfDone(BaseSelector):
        def fit(self, X, y=None):
            return self

    assert 'halfdone' not in BaseSelector._registry
