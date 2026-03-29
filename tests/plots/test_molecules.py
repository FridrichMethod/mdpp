"""Tests for mdpp.plots.molecules."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
from PIL import Image
from rdkit import Chem

from mdpp.plots.molecules import draw_mol, draw_mols


@pytest.fixture()
def ethanol() -> Chem.rdchem.Mol:
    return Chem.MolFromSmiles("CCO")


@pytest.fixture()
def aspirin() -> Chem.rdchem.Mol:
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


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

    def test_highlight_disabled(self, ethanol):
        img = draw_mol(ethanol, highlight=False)
        assert isinstance(img, Image.Image)


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
