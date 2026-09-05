"""Regression tests for reusable FEP+ graph and atom-mapping plots."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.offsetbox import AnnotationBbox
from rdkit import Chem
from rdkit.Chem import rdDepictor

from mdpp.plots import fepp


@pytest.fixture
def mapping_record() -> dict:
    mol_a = Chem.MolFromSmiles("C[NH3+]")
    mol_b = Chem.MolFromSmiles("C=C")
    return {
        "name_a": "A",
        "name_b": "B",
        "similarity": 0.5,
        "molblock_a": Chem.MolToMolBlock(mol_a),
        "molblock_b": Chem.MolToMolBlock(mol_b),
        "core_a": [1, 2],
        "core_b": [1, 2],
        "dummy_a": [],
        "dummy_b": [],
    }


def test_plot_mapping_preserves_inputs_and_exposes_core_changes(mapping_record):
    before = json.dumps(mapping_record, sort_keys=True)
    mol_a, mol_b, map_a, map_b = fepp._aligned_mols(mapping_record)
    changes = fepp._core_changes(mapping_record, mol_a, mol_b, map_a, map_b)
    assert changes == {
        "atoms_a": [1],
        "atoms_b": [1],
        "bonds_a": [0],
        "bonds_b": [0],
        "bond_change_count": 1,
    }
    fig, axes = plt.subplots(1, 2)
    try:
        result = fepp.plot_fep_mapping(mapping_record, ax=axes[0], size=120)
        assert result is axes[0]
        assert len(axes[0].images) == 1
        assert axes[0].images[0].get_array().shape[:2] == (120, 240)
        assert "1 core atoms; 1 core bonds" in axes[0].get_title()
        assert len(axes[1].images) == 0
        assert plt.fignum_exists(fig.number)
        assert json.dumps(mapping_record, sort_keys=True) == before
    finally:
        plt.close(fig)


def test_mapping_hydrogen_removal_preserves_suite_index_correspondence():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    # Put hydrogens before heavy atoms to exercise the 1-based all-atom remap.
    order = [*range(3, mol.GetNumAtoms()), 0, 1, 2]
    mol = Chem.RenumberAtoms(mol, order)
    heavy, index_map = fepp._heavy_mol(Chem.MolToMolBlock(mol))
    assert heavy.GetNumAtoms() == 3
    assert index_map == {len(order) - 3 + i: i for i in range(3)}


def test_alignment_rotates_without_distorting_internal_distances():
    mol_a = Chem.MolFromSmiles("CCO")
    mol_b = Chem.Mol(mol_a)
    rdDepictor.Compute2DCoords(mol_a)
    rdDepictor.Compute2DCoords(mol_b)
    original = mol_b.GetConformer().GetPositions().copy()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for index, point in enumerate(original @ rotation + [5.0, -3.0, 0.0]):
        mol_b.GetConformer().SetAtomPosition(index, point)
    fepp._rigid_align_2d(mol_b, mol_a, [(0, 0), (1, 1), (2, 2)])
    np.testing.assert_allclose(mol_b.GetConformer().GetPositions(), original, atol=1e-12)


@pytest.mark.parametrize("core_b", [[1], [1, 1], [1, 99]])
def test_mapping_rejects_invalid_correspondence(mapping_record, core_b):
    mapping_record["core_b"] = core_b
    with pytest.raises(ValueError, match=r"equal length|unique|out-of-range"):
        fepp.plot_fep_mapping(mapping_record)


def test_save_mappings_writes_outputs_without_leaking_figures(tmp_path, mapping_record):
    before = plt.get_fignums()
    pdf = tmp_path / "pages" / "edges.pdf"
    png = tmp_path / "overview" / "edges.png"
    count = fepp.save_fep_mappings(
        {"fmp": "test", "edges": [mapping_record]}, pdf_path=str(pdf), png_path=png
    )
    assert count == 1
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == before


def test_save_mappings_empty_input_fails_before_writing(tmp_path):
    with pytest.raises(ValueError, match="no edges"):
        fepp.save_fep_mappings(
            {"edges": []}, pdf_path=tmp_path / "a.pdf", png_path=tmp_path / "a.png"
        )
    assert not list(tmp_path.iterdir())


@pytest.fixture
def ligand_dir(tmp_path: Path) -> Path:
    for name, smiles in [("A", "CC"), ("B", "CO"), ("C", "CN")]:
        molecule = Chem.MolFromSmiles(smiles)
        molecule.SetProp("_Name", name)
        Chem.MolToMolFile(molecule, str(tmp_path / f"{name}.sdf"))
    return tmp_path


def test_map_reuses_axes_and_keeps_node_labels_and_directions(ligand_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    try:
        result = fepp.plot_fep_map(
            [("A", "B"), ("B", "C")], ligand_dir=str(ligand_dir), ax=axes[0], title="custom"
        )
        assert result is axes[0]
        assert result.get_title() == "custom"
        assert {text.get_text() for text in result.texts} == {"A", "B", "C"}
        assert len([a for a in result.artists if isinstance(a, AnnotationBbox)]) == 3
        assert len(result.patches) == 2
        assert not axes[1].artists
        assert plt.fignum_exists(fig.number)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([], "at least one"),
        ([("A", "B"), ("C", "D")], "disconnected"),
        ([("A", "missing")], "missing ligand"),
    ],
)
def test_map_rejects_invalid_network(ligand_dir, edges, message):
    with pytest.raises(ValueError, match=message):
        fepp.plot_fep_map(edges, ligand_dir=ligand_dir)


def test_node_depiction_rejects_mismatched_ligand_title(ligand_dir):
    path = ligand_dir / "A.sdf"
    path.write_text(path.read_text().replace("A\n", "wrong\n", 1))
    with pytest.raises(ValueError, match="does not match filename"):
        fepp._ligand_image(path)


def test_spreading_is_deterministic_and_does_not_mutate_layout():
    original = {"A": (-1.0, -1.0), "B": (-0.99, -0.99), "C": (1.0, 1.0)}
    snapshot = dict(original)
    separated = fepp._spread_nodes(original, 0.5)
    assert fepp._min_separation(separated) > fepp._min_separation(original) + 0.4
    assert separated == fepp._spread_nodes(original, 0.5)
    assert original == snapshot


def test_import_does_not_change_matplotlib_backend(monkeypatch):
    def reject_backend_change(*_args, **_kwargs):
        raise AssertionError("library import must not select a plotting backend")

    monkeypatch.setattr(matplotlib, "use", reject_backend_change)
    importlib.reload(fepp)


@pytest.mark.parametrize("caller_axes", [False, True])
def test_bad_ligand_does_not_leak_or_close_figures(ligand_dir, caller_axes):
    path = ligand_dir / "B.sdf"
    path.write_text(path.read_text().replace("B\n", "wrong\n", 1))
    fig, ax = plt.subplots() if caller_axes else (None, None)
    before = plt.get_fignums()
    try:
        with pytest.raises(ValueError, match="does not match filename"):
            fepp.plot_fep_map([("A", "B")], ligand_dir=ligand_dir, ax=ax)
        assert plt.get_fignums() == before
    finally:
        if fig is not None:
            plt.close(fig)
