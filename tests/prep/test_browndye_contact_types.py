"""Tests for ``mdpp.prep.browndye.write_contact_types``."""

from __future__ import annotations

from pathlib import Path

import pytest

from mdpp.prep import write_contact_types


@pytest.fixture()
def two_pqrs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two tiny PQR files plus a target XML path inside ``tmp_path``."""
    mol0 = tmp_path / "mol0.pqr"
    mol0.write_text(
        # Two heavy atoms (CA in ALA, CB in ALA) and one hydrogen that must be excluded.
        # Also include a duplicate heavy atom on residue 2 to confirm dedup.
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  0.00 1.00\n"
        "ATOM      2  CB  ALA A   1       1.000   0.000   0.000  0.00 1.00\n"
        "ATOM      3  HA  ALA A   1       0.500   0.500   0.500  0.00 0.50\n"
        "ATOM      4  CA  ALA A   2       2.000   0.000   0.000  0.00 1.00\n",
        encoding="utf-8",
    )
    mol1 = tmp_path / "mol1.pqr"
    mol1.write_text(
        "ATOM      1  N   GLY B   1       0.000   0.000   0.000  0.00 1.00\n"
        "ATOM      2  H   GLY B   1       0.500   0.500   0.500  0.00 0.50\n",
        encoding="utf-8",
    )
    out = tmp_path / "contact_types.xml"
    return mol0, mol1, out


def test_returns_path_and_writes_file(two_pqrs: tuple[Path, Path, Path]) -> None:
    mol0, mol1, out = two_pqrs
    returned = write_contact_types(mol0, mol1, out)
    assert returned == out
    assert out.is_file()


def test_excludes_hydrogens(two_pqrs: tuple[Path, Path, Path]) -> None:
    mol0, mol1, out = two_pqrs
    write_contact_types(mol0, mol1, out)
    text = out.read_text()
    assert "<atom> HA </atom>" not in text
    assert "<atom> H </atom>" not in text


def test_unique_keys_per_molecule(two_pqrs: tuple[Path, Path, Path]) -> None:
    """Duplicate (atom, residue) pairs collapse; distinct (atom, residue) keys stay."""
    mol0, mol1, out = two_pqrs
    write_contact_types(mol0, mol1, out)
    text = out.read_text()
    # mol0 has CA+ALA appearing twice (res 1 and res 2) but ALA matches both: dedup expected.
    assert text.count("<atom> CA </atom> <residue> ALA </residue>") == 1
    assert text.count("<atom> CB </atom> <residue> ALA </residue>") == 1
    assert text.count("<atom> N </atom> <residue> GLY </residue>") == 1


def test_writes_two_molecule_blocks_in_order(two_pqrs: tuple[Path, Path, Path]) -> None:
    mol0, mol1, out = two_pqrs
    write_contact_types(mol0, mol1, out)
    text = out.read_text()
    mol0_pos = text.find("<molecule0>")
    mol1_pos = text.find("<molecule1>")
    assert mol0_pos > 0
    assert mol1_pos > mol0_pos
    assert "<contacts>" in text and "</contacts>" in text
    assert "<combinations>" in text and "</combinations>" in text
