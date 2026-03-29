"""Tests for mdpp.chem.suppliers."""

from __future__ import annotations

import pytest

from mdpp.chem.suppliers import MolSupplier


class TestMolSupplier:
    def test_smi_file(self, tmp_path):
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol\nc1ccccc1 benzene\n")
        supplier = MolSupplier(str(smi_file))
        mols = list(supplier)
        assert len(mols) == 2

    def test_smi_file_path_object(self, tmp_path):
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol\n")
        supplier = MolSupplier(smi_file)
        mol = next(supplier)
        assert mol is not None

    def test_invalid_format(self):
        with pytest.raises(TypeError, match="Unsupported file format"):
            MolSupplier("molecules.xyz")

    def test_mae_multithreaded_raises(self):
        with pytest.raises(TypeError, match="Multithreading is not supported"):
            MolSupplier("molecules.mae", multithreaded=True)

    def test_skips_empty_molecule(self, tmp_path):
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("INVALID_SMILES\nCCO ethanol\n")
        supplier = MolSupplier(str(smi_file))
        with pytest.warns(RuntimeWarning, match="Empty molecule is skipped"):
            mol = next(supplier)
        assert mol is not None
