"""Tests for mdpp.plots.molecules."""

from __future__ import annotations

from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg")

import pytest
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem

from mdpp.plots import molecules as molecules_module
from mdpp.plots.molecules import draw_mol, draw_mols, get_highlight_bonds


@pytest.fixture()
def ethanol() -> Chem.rdchem.Mol:
    return Chem.MolFromSmiles("CCO")


@pytest.fixture()
def aspirin() -> Chem.rdchem.Mol:
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


def _conformer_positions(mol: Chem.rdchem.Mol) -> list[tuple[float, float, float]]:
    conformer = mol.GetConformer()
    return [
        (
            conformer.GetAtomPosition(atom_idx).x,
            conformer.GetAtomPosition(atom_idx).y,
            conformer.GetAtomPosition(atom_idx).z,
        )
        for atom_idx in range(mol.GetNumAtoms())
    ]


class TestDrawMol:
    def test_returns_image(self, ethanol):
        img = draw_mol(ethanol)
        assert isinstance(img, Image.Image)

    def test_custom_size(self, ethanol):
        img = draw_mol(ethanol, img_size=(400, 400))
        assert isinstance(img, Image.Image)

    def test_with_matching_pattern(self, aspirin):
        pattern = Chem.MolFromSmarts("c1ccccc1")
        img = draw_mol(aspirin, pattern=pattern)
        assert isinstance(img, Image.Image)

    def test_highlight_disabled_with_pattern(self, aspirin):
        pattern = Chem.MolFromSmarts("c1ccccc1")
        img = draw_mol(aspirin, pattern=pattern, highlight=False)
        assert isinstance(img, Image.Image)

    def test_highlight_disabled_still_aligns_to_pattern(self, aspirin, monkeypatch):
        pattern = Chem.MolFromSmarts("c1ccccc1")
        align_mock = Mock(wraps=AllChem.GenerateDepictionMatching2DStructure)
        monkeypatch.setattr(AllChem, "GenerateDepictionMatching2DStructure", align_mock)

        img = draw_mol(aspirin, pattern=pattern, highlight=False)

        assert isinstance(img, Image.Image)
        align_mock.assert_called_once()

    def test_highlight_color_is_applied(self, aspirin, monkeypatch):
        highlight_color = (1.0, 0.0, 0.0, 0.5)
        captured: dict[str, object] = {}

        class _FakeDrawOptions:
            def setHighlightColour(self, color):
                captured["color"] = color

        def _fake_build_draw_options():
            return _FakeDrawOptions()

        monkeypatch.setattr(molecules_module, "build_draw_options", _fake_build_draw_options)

        def _fake_mol_to_image(*_, **__):
            return Image.new("RGB", (10, 10), "white")

        monkeypatch.setattr(molecules_module.Draw, "MolToImage", _fake_mol_to_image)

        img = draw_mol(aspirin, highlight_color=highlight_color)

        assert isinstance(img, Image.Image)
        assert captured["color"] == highlight_color

    def test_does_not_add_conformers_to_input(self, ethanol):
        assert ethanol.GetNumConformers() == 0

        img = draw_mol(ethanol)

        assert isinstance(img, Image.Image)
        assert ethanol.GetNumConformers() == 0

    def test_does_not_modify_existing_coordinates(self, aspirin):
        AllChem.Compute2DCoords(aspirin)
        before_positions = _conformer_positions(aspirin)

        img = draw_mol(aspirin)

        assert isinstance(img, Image.Image)
        assert aspirin.GetNumConformers() == 1
        assert _conformer_positions(aspirin) == before_positions

    def test_does_not_modify_pattern(self, aspirin):
        pattern = Chem.MolFromSmarts("c1ccccc1")
        assert pattern.GetNumConformers() == 0

        img = draw_mol(aspirin, pattern=pattern)

        assert isinstance(img, Image.Image)
        assert pattern.GetNumConformers() == 0


class TestGetHighlightBonds:
    def test_maps_pattern_bonds_to_molecule_indices(self, aspirin):
        pattern = Chem.MolFromSmarts("c1ccccc1")
        atoms = list(aspirin.GetSubstructMatch(pattern))

        bond_indices = get_highlight_bonds(aspirin, atoms, pattern)

        assert len(bond_indices) == pattern.GetNumBonds()
        assert len(set(bond_indices)) == pattern.GetNumBonds()


class TestDrawMols:
    def test_returns_image(self, ethanol, aspirin):
        img = draw_mols([ethanol, aspirin])
        assert isinstance(img, Image.Image)

    def test_with_legends(self, ethanol, aspirin):
        img = draw_mols([ethanol, aspirin], legends=["Ethanol", "Aspirin"])
        assert isinstance(img, Image.Image)

    def test_save_to_file(self, ethanol, aspirin, tmp_path):
        outfile = str(tmp_path / "grid.png")
        img = draw_mols([ethanol, aspirin], output_file=outfile)
        assert isinstance(img, Image.Image)
        assert (tmp_path / "grid.png").exists()

    def test_with_pattern(self, aspirin):
        pattern = Chem.MolFromSmarts("c1ccccc1")
        img = draw_mols([aspirin, aspirin], pattern=pattern)
        assert isinstance(img, Image.Image)

    def test_highlight_disabled_still_aligns_to_pattern(self, aspirin, monkeypatch):
        pattern = Chem.MolFromSmarts("c1ccccc1")
        align_mock = Mock(wraps=AllChem.GenerateDepictionMatching2DStructure)
        monkeypatch.setattr(AllChem, "GenerateDepictionMatching2DStructure", align_mock)

        img = draw_mols([aspirin, aspirin], pattern=pattern, highlight=False)

        assert isinstance(img, Image.Image)
        assert align_mock.call_count == 2

    def test_unmatched_pattern_without_name_is_allowed(self, ethanol):
        pattern = Chem.MolFromSmarts("c1ccccc1")
        img = draw_mols([ethanol], pattern=pattern)
        assert isinstance(img, Image.Image)

    def test_does_not_modify_input_molecules(self, ethanol, aspirin):
        molecules = [ethanol, aspirin]

        assert all(mol.GetNumConformers() == 0 for mol in molecules)

        img = draw_mols(molecules, pattern=Chem.MolFromSmarts("CO"))

        assert isinstance(img, Image.Image)
        assert all(mol.GetNumConformers() == 0 for mol in molecules)
