"""Tests for ``mdpp.prep.apbs.infer_debye_length``."""

from __future__ import annotations

from pathlib import Path

import pytest

from mdpp.prep import infer_debye_length


def test_parses_colon_format(tmp_path: Path) -> None:
    log = tmp_path / "a.log"
    log.write_text(
        "Some preamble.\nDebye length: 12.34 A\nMore output below.\n",
        encoding="utf-8",
    )
    assert infer_debye_length(log) == pytest.approx(12.34)


def test_parses_got_format(tmp_path: Path) -> None:
    log = tmp_path / "b.log"
    log.write_text(
        "Configuring APBS...\ngot debye length 9.99\nComputing potential.\n",
        encoding="utf-8",
    )
    assert infer_debye_length(log) == pytest.approx(9.99)


def test_skips_missing_logs_and_returns_first_match(tmp_path: Path) -> None:
    missing = tmp_path / "absent.log"
    populated = tmp_path / "ok.log"
    populated.write_text("debye length: 7.50 A\n", encoding="utf-8")
    assert infer_debye_length(missing, populated) == pytest.approx(7.50)


def test_rejects_inconsistent_ionic_conditions(tmp_path: Path) -> None:
    first = tmp_path / "first.log"
    first.write_text("Debye length: 1.00 A\n", encoding="utf-8")
    second = tmp_path / "second.log"
    second.write_text("Debye length: 2.00 A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent Debye lengths"):
        infer_debye_length(first, second)


def test_raises_when_no_log_matches(tmp_path: Path) -> None:
    log = tmp_path / "no_debye.log"
    log.write_text("No relevant entries here.\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not infer Debye length"):
        infer_debye_length(log)


def test_raises_when_no_logs_provided() -> None:
    with pytest.raises(RuntimeError, match="Could not infer Debye length"):
        infer_debye_length()


@pytest.mark.parametrize("value", ["7.8500e+00", "7.8500E+00", "7.85"])
def test_parses_scientific_notation(tmp_path: Path, value: str) -> None:
    log = tmp_path / "a.log"
    log.write_text(f"Debye length: {value} A\n")
    assert infer_debye_length(log) == pytest.approx(7.85)


@pytest.mark.parametrize("value", ["inf", "nan", "-1.0", "0.0"])
def test_rejects_nonphysical_debye_length(tmp_path: Path, value: str) -> None:
    log = tmp_path / "a.log"
    log.write_text(f"Debye length: {value} A\n")
    with pytest.raises(ValueError, match="finite and positive"):
        infer_debye_length(log)


def test_allows_log_rounding_and_repeated_focusing_entries(tmp_path: Path) -> None:
    first, second = tmp_path / "a.log", tmp_path / "b.log"
    first.write_text("Debye length: 7.8500 A\nDebye length: 7.8500 A\n")
    second.write_text("Debye length: 7.85 A\n")
    assert infer_debye_length(first, second) == pytest.approx(7.85)
