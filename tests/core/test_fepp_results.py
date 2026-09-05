"""Normalized FEP+ schema and provenance reader regressions."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from mdpp.core import fepp_results as reader


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([("A", "B", 1.0, 0.0)], "greater than zero"),
        ([("A", "A", 1.0, 0.1)], "self-edge"),
    ],
)
def test_invalid_edges_are_rejected(
    tmp_path: Path,
    rows: list[tuple[str, str, float, float]],
    match: str,
    write_edges: Callable[..., None],
) -> None:
    path = tmp_path / "invalid.csv"
    write_edges(path, "open", rows)
    with pytest.raises(ValueError, match=match):
        reader.read_edge_file(path, expected_state="open")


@pytest.mark.parametrize("run_seed", [-1, reader.MAX_SEED + 1])
def test_normalized_seed_outside_fep_plus_domain_is_rejected(
    tmp_path: Path, run_seed: int, write_edges: Callable[..., None]
) -> None:
    path = tmp_path / "invalid_seed.csv"
    write_edges(path, "open", [("A", "B", 1.0, 0.1)], run_seed=run_seed)
    with pytest.raises(ValueError, match=r"run_seed must be in \[0, 2147483647\]"):
        reader.read_edge_file(path, expected_state="open")


def test_blank_vendor_export_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vendor.csv"
    path.write_text("Ligand1,Ligand2,bennett_ddg,bennett_ddg_error\nA,B,,\n")
    with pytest.raises(ValueError, match="blank required field 'bennett_ddg'"):
        reader.read_edge_file(path, expected_state="open")


def test_normalized_commit_marker_and_hash_are_required(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    path = tmp_path / "normalized.csv"
    write_edges(path, "open", [("A", "B", 1.0, 0.1)])
    marker = path.with_suffix(".csv.provenance.json")
    marker.unlink()
    with pytest.raises(ValueError, match="missing commit marker"):
        reader.read_edge_file(path, expected_state="open")

    write_edges(path, "open", [("A", "B", 1.0, 0.1)])
    path.write_text(path.read_text().replace(",1.0,", ",1.1,"))
    with pytest.raises(ValueError, match="normalized result SHA-256 does not match"):
        reader.read_edge_file(path, expected_state="open")

    write_edges(path, "open", [("A", "B", 1.0, 0.1)])
    path.with_suffix(".csv.lock").unlink()
    with pytest.raises(ValueError, match="missing lock"):
        reader.read_edge_file(path, expected_state="open")


def test_normalized_v3_rejects_legacy_commit_marker(
    tmp_path: Path, write_single_edges: Callable[..., None]
) -> None:
    result_csv = tmp_path / "open.csv"
    write_single_edges(result_csv, "open", [("A", "B", 1.0, 0.2)])
    marker_path = result_csv.with_suffix(".csv.provenance.json")
    marker = json.loads(marker_path.read_text())
    marker["schema_version"] = 2
    marker.pop("input_environment_fingerprint")
    marker.pop("completed_environment_fingerprint")
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")

    with pytest.raises(ValueError, match="requires commit-marker schema 3"):
        reader.read_edge_file(result_csv, expected_state="open")

    write_single_edges(result_csv, "open", [("A", "B", 1.0, 0.2)])
    marker = json.loads(marker_path.read_text())
    marker.pop("workflow_type")
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    with pytest.raises(ValueError, match="schema 3 commit marker missing fields"):
        reader.read_edge_file(result_csv, expected_state="open")
