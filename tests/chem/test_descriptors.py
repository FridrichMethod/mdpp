"""Tests for mdpp.chem.descriptors."""

from __future__ import annotations

import pytest

from mdpp.chem.descriptors import COMMON_DESC_NAMES, calc_descs, filt_descs


class TestCalcDescs:
    def test_defaults_returns_tuple(self, ethanol):
        result = calc_descs(ethanol)
        assert isinstance(result, tuple)
        assert len(result) == len(COMMON_DESC_NAMES)

    def test_single_descriptor_returns_float(self, ethanol):
        result = calc_descs(ethanol, desc_names=("MolWt",))
        assert isinstance(result, float)

    def test_aspirin_molwt(self, aspirin):
        mw = calc_descs(aspirin, desc_names=("MolWt",))
        assert mw == pytest.approx(180.16, abs=0.01)

    def test_aspirin_hba_hbd(self, aspirin):
        hba, hbd = calc_descs(aspirin, desc_names=("NumHAcceptors", "NumHDonors"))
        assert hba == 3
        assert hbd == 1

    def test_invalid_descriptor_name(self, ethanol):
        with pytest.raises(KeyError, match="Invalid descriptor names"):
            calc_descs(ethanol, desc_names=("NotADescriptor",))


class TestFiltDescs:
    def test_pass_within_range(self, aspirin):
        filt = {"MolWt": (100.0, 500.0)}
        assert filt_descs(aspirin, filt=filt) is True

    def test_fail_outside_range(self, aspirin):
        filt = {"MolWt": (0.0, 50.0)}
        assert filt_descs(aspirin, filt=filt) is False

    def test_empty_filter_passes(self, aspirin):
        assert filt_descs(aspirin, filt={}) is True

    def test_returns_bool(self, aspirin):
        result = filt_descs(aspirin, filt={"MolWt": (100.0, 500.0)})
        assert type(result) is bool
