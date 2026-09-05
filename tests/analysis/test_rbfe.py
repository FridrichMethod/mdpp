"""Single-state RBFE estimator regressions."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from mdpp.analysis import fepp as analysis
from mdpp.analysis import rbfe as single_analysis


def test_single_state_tree_covariance_pairwise_and_output_bundle(
    tmp_path: Path, write_single_edges: Callable[..., None]
) -> None:
    result_csv = tmp_path / "open.csv"
    write_single_edges(
        result_csv,
        "open",
        [("A", "B", 1.0, 0.2), ("B", "C", 2.0, 0.3)],
    )

    result_bundle = single_analysis.compute_rbfe([result_csv], state="open", reference="A")
    result = result_bundle.summary
    tables = result_bundle.tables
    assert result["quantity"] == "reference_relative_binding_free_energy"
    assert result["state"] == "open"
    np.testing.assert_allclose(result["relative_binding_free_energy"]["value"], [0.0, 1.0, 3.0])
    np.testing.assert_allclose(
        result["relative_binding_free_energy"]["covariance"],
        [[0.0, 0.0, 0.0], [0.0, 0.04, 0.04], [0.0, 0.04, 0.13]],
    )
    pairwise = {(row["source"], row["target"]): row for row in tables["pairwise_contrasts"]}
    assert pairwise[("A", "C")]["target_minus_source"] == pytest.approx(3.0)
    assert pairwise[("A", "C")]["standard_uncertainty"] == pytest.approx(np.sqrt(0.13))
    assert pairwise[("B", "C")]["standard_uncertainty"] == pytest.approx(0.3)

    output_dir = tmp_path / "single_bundle"
    output_dir.mkdir()
    (output_dir / "selectivity.csv").write_text("stale paired result\n")
    single_analysis.write_rbfe(result_bundle, output_dir=output_dir)
    assert {path.name for path in output_dir.iterdir()} == {
        "analysis.json",
        "relative_free_energies.csv",
        "pairwise_contrasts.csv",
        "edge_residuals.csv",
        "cycles.csv",
        "analysis.provenance.json",
    }
    marker = json.loads((output_dir / "analysis.provenance.json").read_text())
    assert set(marker["files"]) == {
        "analysis.json",
        "relative_free_energies.csv",
        "pairwise_contrasts.csv",
        "edge_residuals.csv",
        "cycles.csv",
    }


def test_pairwise_contrasts_are_reference_invariant_and_units_convert(
    tmp_path: Path, write_single_edges: Callable[..., None]
) -> None:
    result_csv = tmp_path / "open.csv"
    write_single_edges(
        result_csv,
        "open",
        [("A", "B", 1.0, 0.2), ("B", "C", 2.0, 0.3)],
    )
    tables_a_bundle = single_analysis.compute_rbfe([result_csv], state="open", reference="A")
    tables_a = tables_a_bundle.tables
    result_b_bundle = single_analysis.compute_rbfe(
        [result_csv], state="open", reference="B", output_unit="kJ/mol"
    )
    result_b = result_b_bundle.summary
    tables_b = result_b_bundle.tables
    contrasts_a = {
        (row["source"], row["target"]): (
            row["target_minus_source"],
            row["standard_uncertainty"],
        )
        for row in tables_a["pairwise_contrasts"]
    }
    contrasts_b = {
        (row["source"], row["target"]): (
            row["target_minus_source"] / 4.184,
            row["standard_uncertainty"] / 4.184,
        )
        for row in tables_b["pairwise_contrasts"]
    }
    assert contrasts_b.keys() == contrasts_a.keys()
    for pair in contrasts_a:
        np.testing.assert_allclose(contrasts_b[pair], contrasts_a[pair])
    np.testing.assert_allclose(
        result_b["relative_binding_free_energy"]["value"], [-4.184, 0.0, 8.368]
    )


def test_single_analysis_rejects_raw_vendor_input_and_mismatched_repeats(
    tmp_path: Path, write_single_edges: Callable[..., None]
) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("Ligand1,Ligand2,bennett_ddg,bennett_ddg_error\nA,B,1.0,0.2\n")
    with pytest.raises(ValueError, match="requires normalized exports"):
        single_analysis.compute_rbfe([raw], state="open", reference="A")

    repeat_a = tmp_path / "repeat_a.csv"
    repeat_b = tmp_path / "repeat_b.csv"
    rows = [("A", "B", 1.0, 0.2)]
    write_single_edges(repeat_a, "open", rows)
    write_single_edges(repeat_b, "open", rows, input_map_sha256="f" * 64)
    with pytest.raises(ValueError, match="repeat input lineage differs"):
        single_analysis.compute_rbfe([repeat_a, repeat_b], state="open", reference="A")


def test_paired_analyzer_rejects_standalone_exports(
    tmp_path: Path, write_single_edges: Callable[..., None]
) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    rows = [("A", "B", 1.0, 0.2)]
    write_single_edges(open_csv, "open", rows)
    write_single_edges(closed_csv, "closed", rows)
    with pytest.raises(ValueError, match="paired-state analysis requires normalized exports"):
        analysis.compute_fep_selectivity([open_csv], closed_paths=[closed_csv], reference="A")


def test_rbfe_library_paths_frozen_result_and_failed_publication(
    tmp_path: Path, write_single_edges: Callable[..., None]
) -> None:
    """The standalone API shares publication safety and accepts string paths."""
    from dataclasses import FrozenInstanceError

    input_path = tmp_path / "open.csv"
    write_single_edges(input_path, "open", [("A", "B", 1.0, 0.2)])
    result = single_analysis.compute_rbfe((str(input_path),), state="open", reference="A")
    assert isinstance(result, single_analysis.RBFEResult)
    with pytest.raises(FrozenInstanceError):
        result.tables = {}  # type: ignore[misc]
    output = tmp_path / "published"
    single_analysis.write_rbfe(result, output_dir=str(output))
    previous = {path.name: path.read_bytes() for path in output.iterdir()}
    malformed = single_analysis.RBFEResult(
        summary={**result.summary, "invalid_value": float("nan")}, tables=result.tables
    )
    with pytest.raises(ValueError, match="Out of range float"):
        single_analysis.write_rbfe(malformed, output_dir=output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous
