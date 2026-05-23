"""Tests for ``mdpp.prep.browndye.build_input_xml`` and ``write_input_xml``."""

from __future__ import annotations

from pathlib import Path

import pytest

from mdpp.prep import BrownDyeBody, BrownDyeSolvent, build_input_xml, write_input_xml
from mdpp.prep.browndye import DEFAULT_BD_SOLVENT_DIELECTRIC


@pytest.fixture()
def bodies() -> tuple[BrownDyeBody, BrownDyeBody]:
    body0 = BrownDyeBody(
        name="complex",
        atoms_xml="complex_atoms.xml",
        grid_dx="complex.dx",
        is_protein=True,
        dielectric=4.0,
    )
    body1 = BrownDyeBody(
        name="substrate",
        atoms_xml="substrate_atoms.xml",
        grid_dx="substrate.dx",
        is_protein=False,
        dielectric=2.5,
    )
    return body0, body1


@pytest.fixture()
def solvent() -> BrownDyeSolvent:
    return BrownDyeSolvent(debye_length_a=8.43)


def test_contains_top_level_tags(
    bodies: tuple[BrownDyeBody, BrownDyeBody], solvent: BrownDyeSolvent
) -> None:
    xml = build_input_xml(*bodies, solvent=solvent)
    for tag in (
        "<top>",
        "</top>",
        "<system>",
        "</system>",
        "<solvent>",
        "<time_step_tolerances>",
    ):
        assert tag in xml


def test_serialises_all_run_knobs(
    bodies: tuple[BrownDyeBody, BrownDyeBody], solvent: BrownDyeSolvent
) -> None:
    xml = build_input_xml(
        *bodies,
        solvent=solvent,
        n_threads=4,
        seed=42,
        n_trajectories=7,
        n_trajectories_per_output=3,
        max_n_steps=999,
        n_steps_per_output=5,
        results_file="custom_results.xml",
        trajectory_file="custom_traj",
        reaction_file="rxn.xml",
    )
    assert "<n_threads> 4 </n_threads>" in xml
    assert "<seed> 42 </seed>" in xml
    assert "<n_trajectories> 7 </n_trajectories>" in xml
    assert "<n_trajectories_per_output> 3 </n_trajectories_per_output>" in xml
    assert "<max_n_steps> 999 </max_n_steps>" in xml
    assert "<n_steps_per_output> 5 </n_steps_per_output>" in xml
    assert "<output> custom_results.xml </output>" in xml
    assert "<trajectory_file> custom_traj </trajectory_file>" in xml
    assert "<reaction_file> rxn.xml </reaction_file>" in xml


def test_serialises_solvent_block(bodies: tuple[BrownDyeBody, BrownDyeBody]) -> None:
    xml = build_input_xml(
        *bodies,
        solvent=BrownDyeSolvent(
            debye_length_a=12.5,
            dielectric=77.0,
            relative_viscosity=1.1,
            kT=1.2,
            desolvation_parameter=0.9,
            solvent_radius_a=1.5,
        ),
    )
    assert "<debye_length> 12.5 </debye_length>" in xml
    assert "<dielectric> 77.0 </dielectric>" in xml
    assert "<relative_viscosity> 1.1 </relative_viscosity>" in xml
    assert "<kT> 1.2 </kT>" in xml
    assert "<desolvation_parameter> 0.9 </desolvation_parameter>" in xml
    assert "<solvent_radius> 1.5 </solvent_radius>" in xml


def test_default_solvent_uses_module_constant(
    bodies: tuple[BrownDyeBody, BrownDyeBody],
) -> None:
    xml = build_input_xml(*bodies, solvent=BrownDyeSolvent(debye_length_a=10.0))
    assert f"<dielectric> {DEFAULT_BD_SOLVENT_DIELECTRIC} </dielectric>" in xml


def test_bool_is_lowercase(
    bodies: tuple[BrownDyeBody, BrownDyeBody], solvent: BrownDyeSolvent
) -> None:
    xml = build_input_xml(*bodies, solvent=solvent)
    assert "<is_protein> true </is_protein>" in xml
    assert "<is_protein> false </is_protein>" in xml
    # all_in_surface defaults to False for both bodies.
    assert xml.count("<all_in_surface> false </all_in_surface>") == 2


def test_body_block_layout(
    bodies: tuple[BrownDyeBody, BrownDyeBody], solvent: BrownDyeSolvent
) -> None:
    xml = build_input_xml(*bodies, solvent=solvent)
    # Two <group> blocks, in body0/body1 order.
    assert xml.count("<group>") == 2
    body0_pos = xml.find("<name> complex </name>")
    body1_pos = xml.find("<name> substrate </name>")
    assert body0_pos > 0 and body1_pos > body0_pos
    assert "<atoms> complex_atoms.xml </atoms>" in xml
    assert "<atoms> substrate_atoms.xml </atoms>" in xml
    assert "<grid> complex.dx </grid>" in xml
    assert "<grid> substrate.dx </grid>" in xml


def test_write_returns_path_and_matches_builder(
    tmp_path: Path,
    bodies: tuple[BrownDyeBody, BrownDyeBody],
    solvent: BrownDyeSolvent,
) -> None:
    out = tmp_path / "input.xml"
    returned = write_input_xml(out, *bodies, solvent=solvent)
    assert returned == out
    assert out.read_text(encoding="utf-8") == build_input_xml(*bodies, solvent=solvent)
