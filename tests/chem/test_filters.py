"""Tests for mdpp.chem.filters."""

from __future__ import annotations

from rdkit import Chem

from mdpp.chem.filters import get_framework, is_pains


class TestGetFramework:
    def test_smiles_input_returns_string(self):
        result = get_framework("c1ccc(CC2CCCCC2)cc1")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mol_input_returns_mol(self):
        mol = Chem.MolFromSmiles("c1ccc(CC2CCCCC2)cc1")
        result = get_framework(mol)
        assert isinstance(result, Chem.rdchem.Mol)

    def test_generic_scaffold(self):
        result = get_framework("c1ccc(CC2CCCCC2)cc1", generic=True)
        assert isinstance(result, str)
        # Generic scaffold has no aromatic atoms -- all single bonds and carbon
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_benzene_scaffold_is_itself(self, benzene):
        scaffold = get_framework(benzene)
        scaffold_smi = Chem.MolToSmiles(scaffold)
        benzene_smi = Chem.MolToSmiles(benzene)
        assert scaffold_smi == benzene_smi


class TestIsPains:
    def test_positive_rhodanine(self, pains_molecule):
        assert is_pains(pains_molecule) is True

    def test_negative_aspirin(self, aspirin):
        assert is_pains(aspirin) is False

    def test_negative_ethanol(self, ethanol):
        assert is_pains(ethanol) is False
