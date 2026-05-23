"""Tests for the frozen+slots invariants on BrownDye config dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from mdpp.prep import BrownDyeBody, BrownDyeSolvent


def test_body_is_frozen() -> None:
    body = BrownDyeBody(name="x", atoms_xml="x_atoms.xml", grid_dx="x.dx")
    with pytest.raises(dataclasses.FrozenInstanceError):
        body.name = "y"  # type: ignore[misc]


def test_solvent_is_frozen() -> None:
    solvent = BrownDyeSolvent(debye_length_a=10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        solvent.debye_length_a = 5.0  # type: ignore[misc]


def test_body_uses_slots() -> None:
    body = BrownDyeBody(name="x", atoms_xml="x_atoms.xml", grid_dx="x.dx")
    assert not hasattr(body, "__dict__")


def test_solvent_uses_slots() -> None:
    solvent = BrownDyeSolvent(debye_length_a=10.0)
    assert not hasattr(solvent, "__dict__")


def test_body_defaults() -> None:
    body = BrownDyeBody(name="x", atoms_xml="a.xml", grid_dx="g.dx")
    assert body.is_protein is True
    assert body.all_in_surface is False
    assert body.dielectric == pytest.approx(4.0)


def test_solvent_defaults() -> None:
    solvent = BrownDyeSolvent(debye_length_a=10.0)
    assert solvent.dielectric == pytest.approx(78.0)
    assert solvent.relative_viscosity == pytest.approx(1.0)
    assert solvent.kT == pytest.approx(1.0)
    assert solvent.desolvation_parameter == pytest.approx(1.0)
    assert solvent.solvent_radius_a == pytest.approx(1.4)
