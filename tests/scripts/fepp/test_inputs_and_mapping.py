"""Regression tests for FEPP molecular inputs and mapping diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from rdkit import Chem

FEPP_DIR = Path(__file__).parents[3] / "examples" / "fepp"


def load_script(name: str) -> ModuleType:
    path = FEPP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ligand_validation = load_script("validate_ligand_inputs")
mapping_plot = load_script("plot_edge_mappings")
map_plot = load_script("plot_fep_map")
microstate_comparison = load_script("compare_receptor_microstates")


def test_repository_ligands_match_smiles_provenance() -> None:
    report = ligand_validation.validate(
        FEPP_DIR / "inputs" / "ligands_amp_fep.smi",
        FEPP_DIR / "inputs" / "ligands",
        expected_charge=-1,
    )
    assert report["status"] == "valid"
    assert report["ligand_count"] == 12
    assert {record["formal_charge"] for record in report["ligands"]} == {-1}


def test_core_atom_and_bond_changes_are_exposed() -> None:
    molecule_a = Chem.MolFromSmiles("C[NH3+]")
    molecule_b = Chem.MolFromSmiles("C=C")
    assert molecule_a is not None and molecule_b is not None
    record = {
        "name_a": "A",
        "name_b": "B",
        "similarity": 0.5,
        "molblock_a": Chem.MolToMolBlock(molecule_a),
        "molblock_b": Chem.MolToMolBlock(molecule_b),
        "core_a": [1, 2],
        "core_b": [1, 2],
        "dummy_a": [],
        "dummy_b": [],
    }
    mol_a, mol_b, map_a, map_b = mapping_plot.aligned_mols(record)
    changes = mapping_plot.core_changes(record, mol_a, mol_b, map_a, map_b)
    assert changes["atoms_a"] == [1]
    assert changes["atoms_b"] == [1]
    assert changes["bonds_a"] == [0]
    assert changes["bonds_b"] == [0]
    assert changes["bond_change_count"] == 1


def test_edge_parser_rejects_malformed_nonblank_lines(tmp_path: Path) -> None:
    edge_file = tmp_path / "bad.edge"
    edge_file.write_text("abc:def  # A -> B\nthis is not an edge\n")
    try:
        map_plot.parse_edges(edge_file)
    except ValueError as exc:
        assert "malformed edge line 2" in str(exc)
    else:
        raise AssertionError("malformed edge line was silently ignored")


def test_microstate_override_never_allows_sequence_mismatch() -> None:
    report = {
        "sequence_mismatches": [{"residue": "A:42"}],
        "composition_mismatches": [],
        "identical_declared_microstates": False,
    }
    try:
        microstate_comparison.enforce_policy(report, allow_microstate_mismatch=True)
    except ValueError as exc:
        assert "sequence mismatches cannot be overridden" in str(exc)
    else:
        raise AssertionError("sequence mismatch was accepted as a microstate override")


def test_microstate_override_never_allows_heavy_atom_composition_mismatch() -> None:
    report = {
        "sequence_mismatches": [],
        "composition_mismatches": [{"residue": "A:42"}],
        "identical_declared_microstates": False,
    }
    try:
        microstate_comparison.enforce_policy(report, allow_microstate_mismatch=True)
    except ValueError as exc:
        assert "composition mismatches cannot be overridden" in str(exc)
    else:
        raise AssertionError("heavy-atom mismatch was accepted as a microstate override")


def test_heavy_atom_topology_detects_cross_residue_rewiring() -> None:
    residue_a = SimpleNamespace(chain="A", resnum=10, inscode="")
    residue_b = SimpleNamespace(chain="A", resnum=20, inscode="")
    neighbor_a = SimpleNamespace(
        pdbname=" SG ",
        element="S",
        getResidue=lambda: residue_a,
    )
    neighbor_b = SimpleNamespace(
        pdbname=" SG ",
        element="S",
        getResidue=lambda: residue_b,
    )
    atom_a = SimpleNamespace(pdbname=" SG ", element="S", bonded_atoms=[neighbor_a])
    atom_b = SimpleNamespace(pdbname=" SG ", element="S", bonded_atoms=[neighbor_b])
    topology_a = microstate_comparison._heavy_atom_topology(atom_a)
    topology_b = microstate_comparison._heavy_atom_topology(atom_b)
    assert topology_a != topology_b
    assert topology_a["bonded_heavy_atoms"][0]["residue"] == "A:10"
    assert topology_b["bonded_heavy_atoms"][0]["residue"] == "A:20"


def test_microstate_override_allows_only_microstate_signature_difference() -> None:
    report = {
        "sequence_mismatches": [],
        "composition_mismatches": [],
        "identical_declared_microstates": False,
    }
    microstate_comparison.enforce_policy(report, allow_microstate_mismatch=True)
