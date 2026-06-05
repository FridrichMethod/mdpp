"""Tests for mdpp.chem.suppliers."""

from __future__ import annotations

import logging

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

    def test_skips_empty_molecule(self, tmp_path, caplog):
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("INVALID_SMILES\nCCO ethanol\n")
        supplier = MolSupplier(str(smi_file))
        with caplog.at_level(logging.WARNING, logger="mdpp.chem.suppliers"):
            mol = next(supplier)
        assert mol is not None
        assert "Empty molecule is skipped." in caplog.text

    def test_context_manager_closes_file_handle(self, tmp_path):
        # The .smr branch opens a text file handle that the supplier must close
        # on context-manager exit (it is the branch with an explicitly tracked
        # handle).
        smr_file = tmp_path / "patterns.smr"
        smr_file.write_text("[#6]\n[#8]\n")
        with MolSupplier(str(smr_file)) as supplier:
            handle = supplier._handles[0]
            mols = list(supplier)
        assert len(mols) == 2
        assert handle.closed

    def test_close_is_idempotent(self, tmp_path):
        smr_file = tmp_path / "patterns.smr"
        smr_file.write_text("[#6]\n[#8]\n")
        supplier = MolSupplier(str(smr_file))
        handle = supplier._handles[0]
        list(supplier)
        supplier.close()
        assert handle.closed
        supplier.close()  # second call must not raise
        assert supplier._handles == []

    def test_close_noop_for_path_supplier(self, tmp_path):
        # The .smi path supplier opens no extra handles; close()/context-manager
        # use must still be safe (no-op).
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol\n")
        with MolSupplier(str(smi_file)) as supplier:
            assert list(supplier)
        supplier.close()
