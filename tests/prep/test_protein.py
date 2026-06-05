"""Tests for ``mdpp.prep.protein`` protonation handling (PROPKA + PDBFixer)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from mdpp.prep.protein import (
    PropkaResidue,
    PropkaResult,
    _propka_variants,
    fix_pdb,
)

# Minimal single-residue glutamate with crude but valid geometry. PDBFixer
# assigns bonds from residue templates, and Modeller adds the GLH proton (HE2)
# only when the GLU variant is forced -- a clean protonation discriminator.
_GLU_PDB = """ATOM      1  N   GLU A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLU A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLU A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   GLU A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  GLU A   1       1.988  -0.773  -1.209  1.00  0.00           C
ATOM      6  CG  GLU A   1       3.508  -0.889  -1.250  1.00  0.00           C
ATOM      7  CD  GLU A   1       4.000  -1.700  -2.430  1.00  0.00           C
ATOM      8  OE1 GLU A   1       3.250  -2.560  -2.930  1.00  0.00           O
ATOM      9  OE2 GLU A   1       5.150  -1.480  -2.850  1.00  0.00           O
ATOM     10  OXT GLU A   1       3.330   1.560   0.000  1.00  0.00           O
TER
END
"""


def _fake_topology(residues: list[tuple[str, str, str]]) -> object:
    """Fake OpenMM topology whose ``residues()`` yields (chain.id, id, name) stubs."""
    objs = [
        SimpleNamespace(chain=SimpleNamespace(id=chain), id=rid, name=name)
        for chain, rid, name in residues
    ]
    return SimpleNamespace(residues=lambda: iter(objs))


@pytest.fixture()
def glu_pdb(tmp_path: Path) -> Path:
    """Write the minimal glutamate fixture and return its path."""
    path = tmp_path / "glu.pdb"
    path.write_text(_GLU_PDB, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# _propka_variants (pure mapping logic)
# --------------------------------------------------------------------------- #
def test_propka_variants_overrides_disagreeing_residues() -> None:
    nonstandard = (
        PropkaResidue("GLU", 54, "A", 7.64, 4.50),  # protonated -> GLH
        PropkaResidue("LYS", 100, "A", 6.00, 10.50),  # deprotonated -> LYN
        PropkaResidue("HIS", 267, "A", 7.50, 6.50),  # protonated -> HIP
    )
    topology = _fake_topology([
        ("A", "53", "ALA"),
        ("A", "54", "GLU"),
        ("A", "100", "LYS"),
        ("A", "267", "HIS"),
    ])
    assert _propka_variants(topology, nonstandard, pH=7.0) == [None, "GLH", "LYN", "HIP"]


def test_propka_variants_cys_deprotonated_to_cyx() -> None:
    nonstandard = (PropkaResidue("CYS", 10, "A", 5.0, 9.0),)  # deprotonated -> CYX
    topology = _fake_topology([("A", "10", "CYS")])
    assert _propka_variants(topology, nonstandard, pH=7.0) == ["CYX"]


def test_propka_variants_skips_unsupported_type(caplog: pytest.LogCaptureFixture) -> None:
    nonstandard = (PropkaResidue("N+", 1, "A", 9.0, 8.0),)  # terminus: no variant
    topology = _fake_topology([("A", "1", "MET")])
    with caplog.at_level(logging.WARNING):
        assert _propka_variants(topology, nonstandard, pH=7.0) == [None]
    assert "no OpenMM variant" in caplog.text


def test_propka_variants_empty_is_all_none() -> None:
    topology = _fake_topology([("A", "1", "ALA"), ("A", "2", "GLY")])
    assert _propka_variants(topology, (), pH=7.0) == [None, None]


# --------------------------------------------------------------------------- #
# fix_pdb dispatch (real PDBFixer + Modeller; PROPKA result is controlled)
# --------------------------------------------------------------------------- #
def _atom_names(pdb_path: Path) -> set[str]:
    return {
        line[12:16].strip()
        for line in pdb_path.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    }


def test_fix_pdb_model_leaves_glu_deprotonated(
    glu_pdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mdpp.prep.protein.run_propka", lambda _p: PropkaResult(()))
    out = tmp_path / "model.pdb"
    fix_pdb(glu_pdb, out)
    names = _atom_names(out)
    assert "H" in names  # hydrogens were added
    assert "HE2" not in names  # GLU stays charged under the model default


def test_fix_pdb_propka_protonates_glu(
    glu_pdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = PropkaResult((PropkaResidue("GLU", 1, "A", 7.64, 4.50),))
    monkeypatch.setattr("mdpp.prep.protein.run_propka", lambda _p: result)
    out = tmp_path / "propka.pdb"
    fix_pdb(glu_pdb, out, protonation="propka")
    assert "HE2" in _atom_names(out)  # PROPKA-driven GLH proton applied


def test_fix_pdb_propka_without_nonstandard_matches_model(
    glu_pdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mdpp.prep.protein.run_propka", lambda _p: PropkaResult(()))
    out = tmp_path / "propka_empty.pdb"
    fix_pdb(glu_pdb, out, protonation="propka")
    assert "HE2" not in _atom_names(out)  # nothing to override -> model default
