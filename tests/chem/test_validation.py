"""Ligand chemistry validation against independent SMILES provenance."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mdpp.chem.validation import (
    normalized_isomeric_smiles,
    read_ligand_templates,
    validate_ligand_inputs,
)

FEPP_DIR = Path(__file__).parents[2] / "examples" / "fepp"


def _write_ligand(directory: Path, smiles: str, *, title: str = "ligand") -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(molecule, randomSeed=42) == 0
    molecule.SetProp("_Name", title)
    with Chem.SDWriter(str(directory / "ligand.sdf")) as writer:
        writer.write(molecule)
    return molecule


def test_repository_ligands_match_smiles_provenance() -> None:
    report = validate_ligand_inputs(
        FEPP_DIR / "inputs" / "ligands_amp_fep.smi",
        FEPP_DIR / "inputs" / "ligands",
        expected_charge=-1,
        ignore_phosphorus_stereo=True,
    )
    assert report["status"] == "valid"
    assert report["ligand_count"] == 12
    assert {record["formal_charge"] for record in report["ligands"]} == {-1}


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "expected header"),
        ("smiles title\n", "no ligand templates"),
        ("smiles title\nC ligand extra\n", "expected SMILES and title"),
        ("smiles title\nC ligand\nN ligand\n", "duplicate ligand title"),
        ("smiles title\nnot-smiles ligand\n", "invalid SMILES"),
    ],
)
def test_read_ligand_templates_rejects_bad_provenance(
    tmp_path: Path, contents: str, message: str
) -> None:
    provenance = tmp_path / "ligands.smi"
    provenance.write_text(contents)
    with pytest.raises(ValueError, match=message):
        read_ligand_templates(str(provenance))


def test_read_ligand_templates_preserves_charge_and_stereo(tmp_path: Path) -> None:
    provenance = tmp_path / "ligands.smi"
    provenance.write_text("smiles title\n\nC[C@H](O)C(=O)[O-] lactate\n")
    template = read_ligand_templates(provenance)["lactate"]
    assert Chem.GetFormalCharge(template) == -1
    assert "@" in Chem.MolToSmiles(template)


def test_normalization_preserves_phosphorus_stereo_by_default() -> None:
    molecule_a = Chem.MolFromSmiles("C[P@](=O)(O)CC")
    molecule_b = Chem.MolFromSmiles("C[P@@](=O)(O)CC")
    before = Chem.MolToSmiles(molecule_a)
    assert normalized_isomeric_smiles(molecule_a) != normalized_isomeric_smiles(molecule_b)
    assert normalized_isomeric_smiles(
        molecule_a, ignore_phosphorus_stereo=True
    ) == normalized_isomeric_smiles(molecule_b, ignore_phosphorus_stereo=True)
    assert Chem.MolToSmiles(molecule_a) == before


def test_3d_inference_recovers_double_bond_geometry_without_mutation(tmp_path: Path) -> None:
    molecule = _write_ligand(tmp_path, "F/C=C/F")
    for bond in molecule.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetStereo(Chem.BondStereo.STEREOANY)
            bond.SetBondDir(Chem.BondDir.EITHERDOUBLE)
    before = Chem.MolToMolBlock(molecule)
    assert normalized_isomeric_smiles(molecule, infer_stereo_3d=True) == "F/C=C/F"
    assert Chem.MolToMolBlock(molecule) == before


def test_3d_inference_requires_a_3d_conformer() -> None:
    with pytest.raises(ValueError, match="requires a 3D conformer"):
        normalized_isomeric_smiles(Chem.MolFromSmiles("C"), infer_stereo_3d=True)


def test_generic_validation_has_no_dataset_charge_default(tmp_path: Path) -> None:
    provenance = tmp_path / "ligands.smi"
    provenance.write_text("smiles title\nC[NH3+] ligand\n")
    _write_ligand(tmp_path, "C[NH3+]")
    report = validate_ligand_inputs(str(provenance), str(tmp_path))
    assert report["expected_formal_charge"] is None
    assert report["ligands"][0]["formal_charge"] == 1
    with pytest.raises(ValueError, match="does not match expected -1"):
        validate_ligand_inputs(provenance, tmp_path, expected_charge=-1)


@pytest.mark.parametrize(
    ("template", "observed", "title", "message"),
    [
        ("CC", "CC", "wrong-name", "does not match filename"),
        ("CC", "CO", "ligand", "chemistry differs"),
        ("F/C=C/F", "F/C=C\\F", "ligand", "chemistry differs"),
        ("C[NH3+]", "CN", "ligand", "chemistry differs"),
        ("C[P@](=O)(O)CC", "C[P@@](=O)(O)CC", "ligand", "chemistry differs"),
    ],
)
def test_validation_rejects_changed_declared_chemistry(
    tmp_path: Path, template: str, observed: str, title: str, message: str
) -> None:
    provenance = tmp_path / "ligands.smi"
    provenance.write_text(f"smiles title\n{template} ligand\n")
    _write_ligand(tmp_path, observed, title=title)
    with pytest.raises(ValueError, match=message):
        validate_ligand_inputs(provenance, tmp_path)


def test_validation_rejects_different_file_set(tmp_path: Path) -> None:
    provenance = tmp_path / "ligands.smi"
    provenance.write_text("smiles title\nCC expected\n")
    _write_ligand(tmp_path, "CC")
    with pytest.raises(ValueError, match="missing=\\['expected'\\], extra=\\['ligand'\\]"):
        validate_ligand_inputs(provenance, tmp_path)


def test_validation_rejects_multiple_records_and_non_3d(tmp_path: Path) -> None:
    provenance = tmp_path / "ligands.smi"
    provenance.write_text("smiles title\nCC ligand\n")
    molecule = _write_ligand(tmp_path, "CC")
    with Chem.SDWriter(str(tmp_path / "ligand.sdf")) as writer:
        writer.write(molecule)
        writer.write(molecule)
    with pytest.raises(ValueError, match="expected exactly one SDF record, got 2"):
        validate_ligand_inputs(provenance, tmp_path)
    molecule.GetConformer().Set3D(False)
    for index in range(molecule.GetNumAtoms()):
        point = molecule.GetConformer().GetAtomPosition(index)
        molecule.GetConformer().SetAtomPosition(index, (point.x, point.y, 0.0))
    with Chem.SDWriter(str(tmp_path / "ligand.sdf")) as writer:
        writer.write(molecule)
    with pytest.raises(ValueError, match="expected exactly one 3D conformer"):
        validate_ligand_inputs(provenance, tmp_path)


def test_example_cli_retains_dataset_defaults(tmp_path: Path) -> None:
    report_path = tmp_path / "nested" / "ligands.json"
    process = subprocess.run(
        [sys.executable, str(FEPP_DIR / "validate_ligand_inputs.py"), "-o", str(report_path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert process.stdout.strip() == "validated 12 ligand inputs"
    assert json.loads(report_path.read_text())["expected_formal_charge"] == -1
