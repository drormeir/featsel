"""
Tests for selector self-registration and the FeatureSelector.create factory.
"""

import pytest

from featsel.selectors import ANOVAFSelector, FeatureSelector

BUILT_IN = ['VarianceThreshold', 'ANOVAFSelector', 'MutualInfoSelector',
            'CorrelationSelector', 'RandomSelector', 'LassoSelector',
            'TreeImportanceSelector']


@pytest.fixture(autouse=True)
def restore_registry():
    """Classes defined inside a test must not leak into the next one."""
    saved = dict(FeatureSelector._registry)
    yield
    FeatureSelector._registry.clear()
    FeatureSelector._registry.update(saved)


class _Stub(FeatureSelector):
    """Concrete enough to register; the tests only check names."""

    def fit(self, X, y=None):
        return self


del FeatureSelector._registry['_stub']  # a test helper, not a selector


def test_built_in_selectors_register_by_class_name():
    """Test that every built-in selector is reachable by its class name."""
    for name in BUILT_IN:
        assert FeatureSelector.lookup(name).__name__ == name


def test_create_is_case_insensitive_and_passes_params():
    """Test that any casing finds the class and parameters reach its constructor."""
    selector = FeatureSelector.create('anovafselector', n_features=5)

    assert isinstance(selector, ANOVAFSelector)
    assert selector.n_features == 5


def test_misspelled_parameter_is_rejected():
    """Test that a typo in a parameter name fails instead of being ignored."""
    with pytest.raises(TypeError, match="n_featurs"):
        FeatureSelector.create('ANOVAFSelector', n_featurs=5)


def test_unknown_name_lists_what_is_available():
    """Test that a typo in the name fails with the valid names."""
    with pytest.raises(ValueError, match="Unknown selector 'anova'.*anovafselector"):
        FeatureSelector.create('anova')


def test_aliases_accept_a_string_or_a_list():
    """Test that aliases register alongside the class name."""
    class OneAlias(_Stub, aliases='one'):
        pass

    class TwoAliases(_Stub, aliases=['two', 'Deux']):
        pass

    assert FeatureSelector.lookup('one') is OneAlias
    assert FeatureSelector.lookup('onealias') is OneAlias
    assert FeatureSelector.lookup('two') is TwoAliases
    assert isinstance(FeatureSelector.create('DEUX'), TwoAliases)


def test_alias_equal_to_the_class_name_is_ignored():
    """Test that repeating the class name as an alias is not a collision."""
    class Echo(_Stub, aliases='ECHO'):
        pass

    assert FeatureSelector.lookup('echo') is Echo


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

    assert FeatureSelector.lookup('family') is Parent
    assert FeatureSelector.lookup('child') is Child


def test_abstract_classes_do_not_register():
    """Test that a class leaving fit() unimplemented stays out."""
    class HalfDone(FeatureSelector):
        pass

    assert 'halfdone' not in FeatureSelector._registry
