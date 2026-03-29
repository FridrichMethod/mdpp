"""Shared fixtures for cheminformatics tests."""

from __future__ import annotations

import pytest
from rdkit import Chem


@pytest.fixture()
def ethanol() -> Chem.rdchem.Mol:
    """Simple molecule for basic tests."""
    return Chem.MolFromSmiles("CCO")


@pytest.fixture()
def aspirin() -> Chem.rdchem.Mol:
    """Drug molecule with known descriptor values."""
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


@pytest.fixture()
def benzene() -> Chem.rdchem.Mol:
    """Aromatic molecule for scaffold tests."""
    return Chem.MolFromSmiles("c1ccccc1")


@pytest.fixture()
def pains_molecule() -> Chem.rdchem.Mol:
    """Known PAINS compound (rhodanine derivative)."""
    return Chem.MolFromSmiles("O=C1CSC(=S)N1")
