"""Paired-state FEP+ estimator regressions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from mdpp.analysis import fepp as analysis
from mdpp.core import fepp_results as reader


def test_inconsistent_triangle_exact_fit_and_cycle() -> None:
    edges = [
        analysis.CombinedEdge("A", "B", 1.0, 1.0, 1, 0.0, 0),
        analysis.CombinedEdge("B", "C", 1.0, 1.0, 1, 0.0, 0),
        analysis.CombinedEdge("A", "C", 3.0, 1.0, 1, 0.0, 0),
    ]
    fit = analysis.compute_fep_network(edges, reference="A")

    np.testing.assert_allclose(fit.relative_free_energy_kcal_mol, [0, 4 / 3, 8 / 3])
    np.testing.assert_allclose(
        fit.covariance_kcal2_mol2,
        np.array([[0, 0, 0], [0, 2 / 3, 1 / 3], [0, 1 / 3, 2 / 3]], dtype=float),
    )
    assert fit.diagnostics["chi_square"] == pytest.approx(1 / 3)
    assert fit.diagnostics["degrees_of_freedom"] == 1
    assert fit.diagnostics["p_value"] == pytest.approx(0.56370286165)
    assert len(fit.cycles) == 1
    assert abs(fit.cycles[0]["closure_kcal_mol"]) == pytest.approx(1.0)
    assert fit.cycles[0]["standard_uncertainty_kcal_mol"] == pytest.approx(np.sqrt(3))


def test_open_closed_double_difference_covariance_and_anchor(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    write_edges(open_csv, "open", [("A", "B", 1, 0.2), ("B", "C", 2, 0.3)])
    write_edges(closed_csv, "closed", [("A", "B", 2, 0.4), ("B", "C", 1, 0.5)])

    result_bundle = analysis.compute_fep_selectivity(
        [open_csv], closed_paths=[closed_csv], reference="A"
    )
    result = result_bundle.summary
    assert result["ligands"] == ["A", "B", "C"]
    np.testing.assert_allclose(result["open"]["relative_free_energy"], [0, 1, 3])
    np.testing.assert_allclose(result["closed"]["relative_free_energy"], [0, 2, 3])
    np.testing.assert_allclose(result["relative_selectivity"]["value"], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(
        result["relative_selectivity"]["covariance"],
        np.array([[0, 0, 0], [0, 0.20, 0.20], [0, 0.20, 0.54]], dtype=float),
    )
    assert result["anchored"] is None

    anchor = tmp_path / "anchor.json"
    anchor.write_text(
        json.dumps({
            "reference": "A",
            "quantity": "binding_free_energy_open_minus_closed",
            "sign_convention": "open_minus_closed",
            "estimate": 0.5,
            "standard_uncertainty": 0.1,
            "unit": "kcal/mol",
        })
    )
    anchored_bundle = analysis.compute_fep_selectivity(
        [open_csv],
        closed_paths=[closed_csv],
        reference="A",
        anchor_path=anchor,
    )
    anchored = anchored_bundle.summary
    np.testing.assert_allclose(anchored["anchored"]["value"], [0.5, -0.5, 0.5])
    assert "the external anchor is independent of both RBFE networks" in anchored["assumptions"]
    np.testing.assert_allclose(
        anchored["anchored"]["covariance"],
        [[0.01, 0.01, 0.01], [0.01, 0.21, 0.21], [0.01, 0.21, 0.55]],
    )

    output_dir = tmp_path / "analysis_bundle"
    output_dir.mkdir()
    (output_dir / "relative_free_energies.csv").write_text("stale single result\n")
    analysis.write_fep_selectivity(anchored_bundle, output_dir=output_dir)
    marker = json.loads((output_dir / "analysis.provenance.json").read_text())
    assert marker["publication_status"] == "complete_commit_marker"
    assert output_dir.with_name("analysis_bundle.lock").is_file()
    assert not (output_dir / "relative_free_energies.csv").exists()
    for filename, digest in marker["files"].items():
        assert hashlib.sha256((output_dir / filename).read_bytes()).hexdigest() == digest


def test_edge_reversal_and_energy_unit_are_invariant(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    open_a = tmp_path / "open_a.csv"
    open_b = tmp_path / "open_b.csv"
    closed = tmp_path / "closed.csv"
    write_edges(open_a, "open", [("A", "B", 1.25, 0.2)])
    write_edges(open_b, "open", [("B", "A", -1.25 * 4.184, 0.2 * 4.184)], unit="kJ/mol")
    write_edges(closed, "closed", [("A", "B", 0.5, 0.3)])

    result_a_bundle = analysis.compute_fep_selectivity(
        [open_a], closed_paths=[closed], reference="A"
    )
    result_a = result_a_bundle.summary
    result_b_bundle = analysis.compute_fep_selectivity(
        [open_b], closed_paths=[closed], reference="A"
    )
    result_b = result_b_bundle.summary
    np.testing.assert_allclose(
        result_a["relative_selectivity"]["value"],
        result_b["relative_selectivity"]["value"],
    )
    np.testing.assert_allclose(
        result_a["relative_selectivity"]["covariance"],
        result_b["relative_selectivity"]["covariance"],
    )


def test_repeat_fixed_effect_and_heterogeneity_are_reported(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    repeat_a = tmp_path / "open_a.csv"
    repeat_b = tmp_path / "open_b.csv"
    write_edges(repeat_a, "open", [("A", "B", 1.0, 1.0)])
    write_edges(repeat_b, "open", [("A", "B", 3.0, 1.0)])
    combined, _ = analysis.combine_repeats([repeat_a, repeat_b], state="open")
    assert len(combined) == 1
    assert combined[0].ddg == pytest.approx(2.0)
    assert combined[0].fixed_effect_uncertainty == pytest.approx(1 / np.sqrt(2))
    assert combined[0].uncertainty == pytest.approx(1.0)
    assert combined[0].uncertainty_scale_factor == pytest.approx(np.sqrt(2))
    assert combined[0].heterogeneity_chi2 == pytest.approx(2.0)
    assert combined[0].heterogeneity_p_value == pytest.approx(0.15729920705)
    fit = analysis.compute_fep_network(combined, reference="A")
    assert fit.diagnostics["repeat_heterogeneity_chi_square"] == pytest.approx(2.0)
    assert fit.diagnostics["repeat_heterogeneity_degrees_of_freedom"] == 1


def test_duplicate_repeat_and_cross_state_content_are_rejected(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    shared_source = "a" * 64
    write_edges(
        open_csv,
        "open",
        [("A", "B", 1.0, 0.2)],
        source_fmp_sha256=shared_source,
    )
    write_edges(
        closed_csv,
        "closed",
        [("A", "B", 1.0, 0.2)],
        source_fmp_sha256=shared_source,
    )
    with pytest.raises(ValueError, match="same open result path"):
        analysis.combine_repeats([open_csv, open_csv], state="open")

    with pytest.raises(ValueError, match="same source FMP"):
        analysis.compute_fep_selectivity([open_csv], closed_paths=[closed_csv], reference="A")


def test_every_repeat_edge_set_and_protocol_are_validated(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    first = tmp_path / "first.csv"
    middle = tmp_path / "middle.csv"
    last = tmp_path / "last.csv"
    expected = [("A", "B", 1.0, 0.2), ("B", "C", 2.0, 0.2)]
    write_edges(first, "open", expected)
    write_edges(middle, "open", [("A", "B", 1.0, 0.2), ("A", "C", 3.0, 0.2)])
    write_edges(last, "open", expected)
    with pytest.raises(ValueError, match="repeat edge set differs"):
        analysis.combine_repeats([first, middle, last], state="open")

    incompatible = tmp_path / "incompatible.csv"
    write_edges(incompatible, "open", expected, protocol={"forcefield": "OPLS5"})
    with pytest.raises(ValueError, match="multiple protocol fingerprints"):
        analysis.combine_repeats([first, incompatible], state="open")

    different_map = tmp_path / "different_map.csv"
    write_edges(different_map, "open", expected, input_map_sha256="f" * 64)
    with pytest.raises(ValueError, match="repeat input lineage differs"):
        analysis.combine_repeats([first, different_map], state="open")


def test_mixed_engines_are_rejected_across_states(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    write_edges(open_csv, "open", [("A", "B", 1.0, 0.2)], engine="fep_plus")
    write_edges(closed_csv, "closed", [("A", "B", 1.0, 0.2)], engine="openfe")
    with pytest.raises(ValueError, match="result engines differ"):
        analysis.compute_fep_selectivity([open_csv], closed_paths=[closed_csv], reference="A")


def test_mixed_paired_input_generations_are_rejected(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    write_edges(open_csv, "open", [("A", "B", 1.0, 0.2)])
    write_edges(
        closed_csv,
        "closed",
        [("A", "B", 1.0, 0.2)],
        paired_inputs_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="incompatible input_paired_inputs_sha256"):
        analysis.compute_fep_selectivity([open_csv], closed_paths=[closed_csv], reference="A")


def test_vendor_qc_warnings_are_machine_visible(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    write_edges(open_csv, "open", [("A", "B", 1.0, 0.2)], qc_rating="Fair")
    write_edges(closed_csv, "closed", [("A", "B", 1.0, 0.2)])
    result_bundle = analysis.compute_fep_selectivity(
        [open_csv], closed_paths=[closed_csv], reference="A"
    )
    result = result_bundle.summary
    assert result["quality_control"]["has_warnings"] is True
    assert result["quality_control"]["warning_edge_ratings_across_runs"] == 1
    assert result["input_provenance"]["open"][0]["quality_control"]["warning_edge_count"] == 1


def test_microstate_mismatch_policy_is_validated_and_reported(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    policy = {"input_allow_microstate_mismatch": "1"}
    write_edges(open_csv, "open", [("A", "B", 1.0, 0.2)], protocol=policy)
    write_edges(closed_csv, "closed", [("A", "B", 1.0, 0.2)], protocol=policy)
    result_bundle = analysis.compute_fep_selectivity(
        [open_csv], closed_paths=[closed_csv], reference="A"
    )
    result = result_bundle.summary
    assert "the receptor microstate mismatch allowance was enabled" in result["assumptions"]

    invalid = tmp_path / "invalid_policy.csv"
    write_edges(
        invalid,
        "open",
        [("A", "B", 1.0, 0.2)],
        protocol={"input_allow_microstate_mismatch": "2"},
    )
    with pytest.raises(ValueError, match="unsupported protocol input_allow_microstate_mismatch"):
        reader.read_edge_file(invalid, expected_state="open")


def test_library_results_accept_string_paths_and_are_frozen(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    """Library clients need no example imports or pathlib-only paths."""
    from dataclasses import FrozenInstanceError

    open_path = tmp_path / "open.csv"
    closed_path = tmp_path / "closed.csv"
    write_edges(open_path, "open", [("A", "B", 1.0, 0.2)])
    write_edges(closed_path, "closed", [("A", "B", 2.0, 0.3)])
    result = analysis.compute_fep_selectivity(
        (str(open_path),), closed_paths=(str(closed_path),), reference="A"
    )
    assert isinstance(result, analysis.FEPSelectivityResult)
    np.testing.assert_allclose(result.summary["relative_selectivity"]["value"], [0.0, -1.0])
    with pytest.raises(FrozenInstanceError):
        result.summary = {}  # type: ignore[misc]
    combined, _ = analysis.combine_repeats((str(open_path),), state="open")
    fit = analysis.compute_fep_network(combined, reference="A")
    assert isinstance(fit, analysis.FEPNetworkResult)
    with pytest.raises(FrozenInstanceError):
        fit.reference = "B"  # type: ignore[misc]
    analysis.write_fep_selectivity(result, output_dir=str(tmp_path / "output"))
    assert (tmp_path / "output" / "selectivity.csv").is_file()


def test_failed_paired_publication_preserves_committed_bundle(
    tmp_path: Path, write_edges: Callable[..., None]
) -> None:
    """Serialization failure must not invalidate a previously published result."""
    open_path, closed_path = tmp_path / "open.csv", tmp_path / "closed.csv"
    write_edges(open_path, "open", [("A", "B", 1.0, 0.2)])
    write_edges(closed_path, "closed", [("A", "B", 2.0, 0.3)])
    result = analysis.compute_fep_selectivity(
        [open_path], closed_paths=[closed_path], reference="A"
    )
    output = tmp_path / "published"
    analysis.write_fep_selectivity(result, output_dir=output)
    previous = {path.name: path.read_bytes() for path in output.iterdir()}
    malformed = analysis.FEPSelectivityResult(
        summary={**result.summary, "invalid_value": float("nan")}, tables=result.tables
    )
    with pytest.raises(ValueError, match="Out of range float"):
        analysis.write_fep_selectivity(malformed, output_dir=output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous
