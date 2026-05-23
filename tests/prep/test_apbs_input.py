"""Tests for ``mdpp.prep.apbs.write_apbs_input``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mdpp.prep import write_apbs_input
from mdpp.prep.apbs import (
    _CL_PAULING_RADIUS_A,
    _NA_PAULING_RADIUS_A,
    DEFAULT_FINE_PADDING_A,
    DEFAULT_FINE_SPACING_A,
    DEFAULT_IONIC_STRENGTH_M,
    _apbs_friendly_dime,
)

_PQR_HEADER = "REMARK   minimal PQR fixture\n"

# Three atoms at the corners of a 10 A box with 1 A radii.
_PQR_BODY = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  0.00 1.00\n"
    "ATOM      2  CB  ALA A   1      10.000   0.000   0.000  0.00 1.00\n"
    "ATOM      3  CG  ALA A   1       0.000  10.000  10.000  0.00 1.00\n"
)


@pytest.fixture()
def stem_in_tmp(tmp_path: Path) -> tuple[str, Path]:
    """Write a tiny PQR file to ``tmp_path/foo.pqr`` and return ``("foo", tmp_path)``."""
    (tmp_path / "foo.pqr").write_text(_PQR_HEADER + _PQR_BODY, encoding="utf-8")
    return "foo", tmp_path


def test_writes_input_file_next_to_pqr(stem_in_tmp: tuple[str, Path]) -> None:
    stem, work = stem_in_tmp
    out = write_apbs_input(stem, work)
    assert out == work / f"{stem}.in"
    assert out.is_file()


def test_default_grid_and_ion_lines(stem_in_tmp: tuple[str, Path]) -> None:
    stem, work = stem_in_tmp
    out = write_apbs_input(stem, work)
    text = out.read_text()

    assert text.count(f"mol pqr {stem}.pqr") == 1
    assert (
        f"ion charge -1.00 conc {DEFAULT_IONIC_STRENGTH_M:.4f} "
        f"radius {_CL_PAULING_RADIUS_A:.4f}" in text
    )
    assert (
        f"ion charge 1.00 conc {DEFAULT_IONIC_STRENGTH_M:.4f} "
        f"radius {_NA_PAULING_RADIUS_A:.4f}" in text
    )
    assert "lpbe" in text and "bcfl sdh" in text and "srfm smol" in text


def test_dime_is_apbs_friendly(stem_in_tmp: tuple[str, Path]) -> None:
    """Every ``dime`` triple value must equal some ``c * 2**n + 1``."""
    stem, work = stem_in_tmp
    out = write_apbs_input(stem, work)
    match = re.search(r"^\s*dime\s+(\d+)\s+(\d+)\s+(\d+)", out.read_text(), re.MULTILINE)
    assert match is not None
    candidates = {c * 2**n + 1 for c in range(1, 7) for n in range(1, 12)}
    for value in match.groups():
        assert int(value) in candidates


def test_fglen_honours_fine_padding(stem_in_tmp: tuple[str, Path]) -> None:
    """Fglen = (atom span + 2*radius) + fine_padding per axis."""
    stem, work = stem_in_tmp
    out = write_apbs_input(stem, work)
    text = out.read_text()
    match = re.search(r"^\s*fglen\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text, re.MULTILINE)
    assert match is not None
    fglen = [float(v) for v in match.groups()]
    # X span: [0-1, 10+1] = 12. Y / Z spans: [0-1, 10+1] = 12.
    expected = 12.0 + DEFAULT_FINE_PADDING_A
    assert fglen == pytest.approx([expected, expected, expected], abs=1e-3)


def test_kwarg_overrides_ionic_strength(stem_in_tmp: tuple[str, Path]) -> None:
    stem, work = stem_in_tmp
    out = write_apbs_input(stem, work, ionic_strength_m=0.0500)
    text = out.read_text()
    assert "conc 0.0500" in text
    assert f"conc {DEFAULT_IONIC_STRENGTH_M:.4f}" not in text


def test_kwarg_overrides_dielectric_and_temperature(stem_in_tmp: tuple[str, Path]) -> None:
    stem, work = stem_in_tmp
    out = write_apbs_input(
        stem,
        work,
        solute_dielectric=4.0,
        solvent_dielectric=80.0,
        temperature_k=310.15,
    )
    text = out.read_text()
    assert "pdie 4.0000" in text
    assert "sdie 80.0000" in text
    assert "temp 310.15" in text


def test_raises_on_empty_pqr(tmp_path: Path) -> None:
    (tmp_path / "empty.pqr").write_text("REMARK only header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No ATOM/HETATM records"):
        write_apbs_input("empty", tmp_path)


def test_raises_on_missing_pqr(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_apbs_input("nonexistent", tmp_path)


def test_apbs_friendly_dime_rounds_up() -> None:
    """``_apbs_friendly_dime(length, spacing)`` rounds to next c*2**n+1."""
    # 100 / 0.75 = 133.33 -> ceil + 1 = 135; smallest c*2**n+1 >= 135 is 161 (5*2**5+1).
    assert _apbs_friendly_dime(100.0, DEFAULT_FINE_SPACING_A) == 161
    # 1 / 1 -> target = 2; smallest candidate >= 2 is 3 (1*2**1+1).
    assert _apbs_friendly_dime(1.0, 1.0) == 3
