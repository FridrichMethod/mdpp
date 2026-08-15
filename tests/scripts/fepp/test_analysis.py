"""Regression tests for the paired-state FEP+ network estimator."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

FEPP_DIR = Path(__file__).parents[3] / "examples" / "fepp"
EDGE_HASH = "1" * 64
MAPPING_FINGERPRINT = "2" * 64
LIGAND_BUNDLE_HASH = "3" * 64
LIGAND_VALIDATION_HASH = "4" * 64
MICROSTATE_REPORT_HASH = "5" * 64
PAIRED_INPUTS_HASH = "6" * 64
TEST_PROTOCOL = {
    "suite_release": "Suite TEST",
    "forcefield": "OPLS4",
    "custom_charge_mode": "assign",
    "water": "SPC",
    "ensemble": "muVT",
    "time_ps": "5000",
    "equilibration_time_ps": "20",
    "lambda_windows": "12",
    "salt_molar": "0.0",
    "input_his267_state": "HIE",
    "input_prep_ph": "7.0",
    "input_prep_rmsd_A": "0.3",
    "input_map_topology": "normal",
    "input_allow_microstate_mismatch": "0",
    "edge_sha256": EDGE_HASH,
    "atom_mapping_fingerprint": MAPPING_FINGERPRINT,
    "ligand_bundle_sha256": LIGAND_BUNDLE_HASH,
    "ligand_validation_sha256": LIGAND_VALIDATION_HASH,
    "receptor_microstates_sha256": MICROSTATE_REPORT_HASH,
    "paired_inputs_sha256": PAIRED_INPUTS_HASH,
}


def load_script(name: str) -> ModuleType:
    path = FEPP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = load_script("analyze_fep_results")


def write_edges(
    path: Path,
    state: str,
    rows: list[tuple[str, str, float, float]],
    *,
    unit: str = "kcal/mol",
    engine: str = "fep_plus",
    protocol: dict[str, str] | None = None,
    run_seed: int | None = None,
    run_manifest_sha256: str | None = None,
    run_id: str | None = None,
    source_fmp_sha256: str | None = None,
    input_map_sha256: str | None = None,
    paired_inputs_sha256: str | None = None,
    qc_rating: str = "Good",
) -> None:
    identity = hashlib.sha256(str(path).encode()).hexdigest()
    protocol_fields = dict(TEST_PROTOCOL)
    if protocol is not None:
        protocol_fields.update(protocol)
    if paired_inputs_sha256 is not None:
        protocol_fields["paired_inputs_sha256"] = paired_inputs_sha256
    protocol_json = json.dumps(
        protocol_fields,
        sort_keys=True,
        separators=(",", ":"),
    )
    fieldnames = [
        "schema_version",
        "engine",
        "protocol_fingerprint",
        "protocol_json",
        "run_manifest_sha256",
        "run_id",
        "run_seed",
        "source_fmp_sha256",
        *analysis.INPUT_LINEAGE_FIELDS,
        "vendor_edge_table_sha256",
        *analysis.QC_FIELDS,
        "state",
        "edge_id",
        "source",
        "target",
        "ddg",
        "standard_uncertainty",
        "unit",
        "sign_convention",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (source, target, ddg, uncertainty) in enumerate(rows):
            writer.writerow({
                "schema_version": 2,
                "engine": engine,
                "protocol_fingerprint": hashlib.sha256(protocol_json.encode()).hexdigest(),
                "protocol_json": protocol_json,
                "run_manifest_sha256": run_manifest_sha256 or identity,
                "run_id": run_id or path.stem,
                "run_seed": (
                    run_seed
                    if run_seed is not None
                    else int(identity[:8], 16) % (analysis.MAX_SEED + 1)
                ),
                "source_fmp_sha256": source_fmp_sha256
                or hashlib.sha256(f"source-fmp:{path}".encode()).hexdigest(),
                "input_map_sha256": input_map_sha256
                or hashlib.sha256(f"input-map:{state}".encode()).hexdigest(),
                "input_edge_sha256": EDGE_HASH,
                "input_map_provenance_sha256": hashlib.sha256(
                    f"map-provenance:{state}".encode()
                ).hexdigest(),
                "input_atom_mapping_fingerprint": MAPPING_FINGERPRINT,
                "input_atom_mapping_json_sha256": hashlib.sha256(
                    f"mapping-json:{state}".encode()
                ).hexdigest(),
                "input_ligand_bundle_sha256": LIGAND_BUNDLE_HASH,
                "input_ligand_validation_sha256": LIGAND_VALIDATION_HASH,
                "input_receptor_microstates_sha256": MICROSTATE_REPORT_HASH,
                "input_paired_inputs_sha256": paired_inputs_sha256 or PAIRED_INPUTS_HASH,
                "vendor_edge_table_sha256": hashlib.sha256(
                    f"vendor-edges:{path}".encode()
                ).hexdigest(),
                **dict.fromkeys(analysis.QC_FIELDS, qc_rating),
                "state": state,
                "edge_id": f"e{index:03d}",
                "source": source,
                "target": target,
                "ddg": ddg,
                "standard_uncertainty": uncertainty,
                "unit": unit,
                "sign_convention": "target_minus_source",
            })
    with path.open(newline="") as handle:
        first_row = next(csv.DictReader(handle))
    input_lineage = {field: first_row[field] for field in analysis.INPUT_LINEAGE_FIELDS}
    publication = {
        "schema_version": 2,
        "publication_status": "complete_commit_marker",
        "normalized_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_manifest_sha256": first_row["run_manifest_sha256"],
        "run_id": first_row["run_id"],
        "run_seed": int(first_row["run_seed"]),
        "source_fmp_sha256": first_row["source_fmp_sha256"],
        "state": first_row["state"],
        "protocol_fingerprint": first_row["protocol_fingerprint"],
        "scientific_protocol": json.loads(first_row["protocol_json"]),
        "input_lineage": input_lineage,
        "completed_atom_mapping_fingerprint": input_lineage["input_atom_mapping_fingerprint"],
    }
    path.with_suffix(f"{path.suffix}.lock").touch()
    path.with_suffix(f"{path.suffix}.provenance.json").write_text(
        json.dumps(publication, indent=2) + "\n"
    )


def test_inconsistent_triangle_exact_fit_and_cycle() -> None:
    edges = [
        analysis.CombinedEdge("A", "B", 1.0, 1.0, 1, 0.0, 0),
        analysis.CombinedEdge("B", "C", 1.0, 1.0, 1, 0.0, 0),
        analysis.CombinedEdge("A", "C", 3.0, 1.0, 1, 0.0, 0),
    ]
    fit = analysis.fit_network(edges, reference="A")

    np.testing.assert_allclose(fit["relative_free_energy_kcal_mol"], [0, 4 / 3, 8 / 3])
    np.testing.assert_allclose(
        fit["covariance_kcal2_mol2"],
        np.array([[0, 0, 0], [0, 2 / 3, 1 / 3], [0, 1 / 3, 2 / 3]], dtype=float),
    )
    assert fit["diagnostics"]["chi_square"] == pytest.approx(1 / 3)
    assert fit["diagnostics"]["degrees_of_freedom"] == 1
    assert fit["diagnostics"]["p_value"] == pytest.approx(0.56370286165)
    assert len(fit["cycles"]) == 1
    assert abs(fit["cycles"][0]["closure_kcal_mol"]) == pytest.approx(1.0)
    assert fit["cycles"][0]["standard_uncertainty_kcal_mol"] == pytest.approx(np.sqrt(3))


def test_open_closed_double_difference_covariance_and_anchor(tmp_path: Path) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    write_edges(open_csv, "open", [("A", "B", 1, 0.2), ("B", "C", 2, 0.3)])
    write_edges(closed_csv, "closed", [("A", "B", 2, 0.4), ("B", "C", 1, 0.5)])

    result, _ = analysis.analyze([open_csv], [closed_csv], reference="A")
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
    anchored, anchored_tables = analysis.analyze(
        [open_csv],
        [closed_csv],
        reference="A",
        anchor_path=anchor,
    )
    np.testing.assert_allclose(anchored["anchored"]["value"], [0.5, -0.5, 0.5])
    assert "the external anchor is independent of both RBFE networks" in anchored["assumptions"]
    np.testing.assert_allclose(
        anchored["anchored"]["covariance"],
        [[0.01, 0.01, 0.01], [0.01, 0.21, 0.21], [0.01, 0.21, 0.55]],
    )

    output_dir = tmp_path / "analysis_bundle"
    output_dir.mkdir()
    (output_dir / "relative_free_energies.csv").write_text("stale single result\n")
    analysis.write_outputs(anchored, anchored_tables, output_dir=output_dir)
    marker = json.loads((output_dir / "analysis.provenance.json").read_text())
    assert marker["publication_status"] == "complete_commit_marker"
    assert output_dir.with_name("analysis_bundle.lock").is_file()
    assert not (output_dir / "relative_free_energies.csv").exists()
    for filename, digest in marker["files"].items():
        assert hashlib.sha256((output_dir / filename).read_bytes()).hexdigest() == digest


def test_edge_reversal_and_energy_unit_are_invariant(tmp_path: Path) -> None:
    open_a = tmp_path / "open_a.csv"
    open_b = tmp_path / "open_b.csv"
    closed = tmp_path / "closed.csv"
    write_edges(open_a, "open", [("A", "B", 1.25, 0.2)])
    write_edges(open_b, "open", [("B", "A", -1.25 * 4.184, 0.2 * 4.184)], unit="kJ/mol")
    write_edges(closed, "closed", [("A", "B", 0.5, 0.3)])

    result_a, _ = analysis.analyze([open_a], [closed], reference="A")
    result_b, _ = analysis.analyze([open_b], [closed], reference="A")
    np.testing.assert_allclose(
        result_a["relative_selectivity"]["value"],
        result_b["relative_selectivity"]["value"],
    )
    np.testing.assert_allclose(
        result_a["relative_selectivity"]["covariance"],
        result_b["relative_selectivity"]["covariance"],
    )


def test_repeat_fixed_effect_and_heterogeneity_are_reported(tmp_path: Path) -> None:
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
    fit = analysis.fit_network(combined, reference="A")
    assert fit["diagnostics"]["repeat_heterogeneity_chi_square"] == pytest.approx(2.0)
    assert fit["diagnostics"]["repeat_heterogeneity_degrees_of_freedom"] == 1


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
) -> None:
    path = tmp_path / "invalid.csv"
    write_edges(path, "open", rows)
    with pytest.raises(ValueError, match=match):
        analysis.read_edge_file(path, expected_state="open")


@pytest.mark.parametrize("run_seed", [-1, analysis.MAX_SEED + 1])
def test_normalized_seed_outside_fep_plus_domain_is_rejected(
    tmp_path: Path,
    run_seed: int,
) -> None:
    path = tmp_path / "invalid_seed.csv"
    write_edges(path, "open", [("A", "B", 1.0, 0.1)], run_seed=run_seed)
    with pytest.raises(ValueError, match=r"run_seed must be in \[0, 2147483647\]"):
        analysis.read_edge_file(path, expected_state="open")


def test_blank_vendor_export_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vendor.csv"
    path.write_text("Ligand1,Ligand2,bennett_ddg,bennett_ddg_error\nA,B,,\n")
    with pytest.raises(ValueError, match="blank required field 'bennett_ddg'"):
        analysis.read_edge_file(path, expected_state="open")


def test_normalized_commit_marker_and_hash_are_required(tmp_path: Path) -> None:
    path = tmp_path / "normalized.csv"
    write_edges(path, "open", [("A", "B", 1.0, 0.1)])
    marker = path.with_suffix(".csv.provenance.json")
    marker.unlink()
    with pytest.raises(ValueError, match="missing commit marker"):
        analysis.read_edge_file(path, expected_state="open")

    write_edges(path, "open", [("A", "B", 1.0, 0.1)])
    path.write_text(path.read_text().replace(",1.0,", ",1.1,"))
    with pytest.raises(ValueError, match="normalized result SHA-256 does not match"):
        analysis.read_edge_file(path, expected_state="open")

    write_edges(path, "open", [("A", "B", 1.0, 0.1)])
    path.with_suffix(".csv.lock").unlink()
    with pytest.raises(ValueError, match="missing lock"):
        analysis.read_edge_file(path, expected_state="open")


def test_duplicate_repeat_and_cross_state_content_are_rejected(tmp_path: Path) -> None:
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
        analysis.analyze([open_csv], [closed_csv], reference="A")


def test_every_repeat_edge_set_and_protocol_are_validated(tmp_path: Path) -> None:
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


def test_mixed_engines_are_rejected_across_states(tmp_path: Path) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    write_edges(open_csv, "open", [("A", "B", 1.0, 0.2)], engine="fep_plus")
    write_edges(closed_csv, "closed", [("A", "B", 1.0, 0.2)], engine="openfe")
    with pytest.raises(ValueError, match="result engines differ"):
        analysis.analyze([open_csv], [closed_csv], reference="A")


def test_mixed_paired_input_generations_are_rejected(tmp_path: Path) -> None:
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
        analysis.analyze([open_csv], [closed_csv], reference="A")


def test_vendor_qc_warnings_are_machine_visible(tmp_path: Path) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    write_edges(open_csv, "open", [("A", "B", 1.0, 0.2)], qc_rating="Fair")
    write_edges(closed_csv, "closed", [("A", "B", 1.0, 0.2)])
    result, _ = analysis.analyze([open_csv], [closed_csv], reference="A")
    assert result["quality_control"]["has_warnings"] is True
    assert result["quality_control"]["warning_edge_ratings_across_runs"] == 1
    assert result["input_provenance"]["open"][0]["quality_control"]["warning_edge_count"] == 1


def test_microstate_mismatch_policy_is_validated_and_reported(tmp_path: Path) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    policy = {"input_allow_microstate_mismatch": "1"}
    write_edges(open_csv, "open", [("A", "B", 1.0, 0.2)], protocol=policy)
    write_edges(closed_csv, "closed", [("A", "B", 1.0, 0.2)], protocol=policy)
    result, _ = analysis.analyze([open_csv], [closed_csv], reference="A")
    assert "the receptor microstate mismatch allowance was enabled" in result["assumptions"]

    invalid = tmp_path / "invalid_policy.csv"
    write_edges(
        invalid,
        "open",
        [("A", "B", 1.0, 0.2)],
        protocol={"input_allow_microstate_mismatch": "2"},
    )
    with pytest.raises(ValueError, match="unsupported protocol input_allow_microstate_mismatch"):
        analysis.read_edge_file(invalid, expected_state="open")


def test_vendor_adapter_normalizes_results_and_rejects_blank_export(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "version.txt").write_text("Suite TEST\n")
    runner = suite / "run"
    runner.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == "python3" && "$(basename "${2-}")" == "extract_edge_mappings.py" ]]; then
    shift 2
    output=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f) shift 2 ;;
            -o) output="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    fingerprint="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    environment="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    if [[ "${FAKE_WRONG_MAPPING:-0}" == "1" && "${output}" == *completed_mappings.json ]]; then
        fingerprint="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    fi
    if [[ "${FAKE_WRONG_ENVIRONMENT:-0}" == "1" && "${output}" == *completed_mappings.json ]]; then
        environment="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    fi
    printf '{"schema_version":2,"mapping_fingerprint":"%s","environment_fingerprint":"%s","edges":[]}\n' \
        "${fingerprint}" "${environment}" >"${output}"
    exit 0
fi
output=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        output="$2"
        shift 2
    else
        shift
    fi
done
if [[ "${FAKE_BLANK:-0}" == "1" ]]; then
    printf 'Ligand1,Ligand2,bennett_ddg,bennett_ddg_error,convergence,ligand RMSD,REST exchange,CCC Conv\nA,B,,,Good,Good,Good,Good\n' >"${output}_ddG.csv"
elif [[ "${FAKE_WRONG_EDGE:-0}" == "1" ]]; then
    printf 'Ligand1,Ligand2,bennett_ddg,bennett_ddg_error,convergence,ligand RMSD,REST exchange,CCC Conv\nA,C,1.25,0.2,Good,Good,Good,Good\n' >"${output}_ddG.csv"
else
    printf 'Ligand1,Ligand2,bennett_ddg,bennett_ddg_error,convergence,ligand RMSD,REST exchange,CCC Conv\nB,A,-1.25,0.2,Good,Good,Good,Good\n' >"${output}_ddG.csv"
fi
"""
    )
    runner.chmod(0o755)
    fmp = tmp_path / "completed_out.fmp"
    fmp.write_text("synthetic completed map\n")
    mapping_fingerprint = "a" * 64
    snapshot_contents = {
        "input_map.fmp": "input map\n",
        "input_map.edge": "abc:def  # A -> B\n",
        "input_map.provenance": "map provenance\n",
        "input_map_mappings.json": json.dumps({
            "schema_version": 2,
            "mapping_fingerprint": mapping_fingerprint,
            "edges": [{"name_a": "A", "name_b": "B"}],
        }),
        "input_ligands.maegz": "ligand bundle\n",
        "input_ligand_validation.json": '{"status":"valid"}\n',
        "input_receptor_microstates.json": '{"identical":true}\n',
    }
    for filename, content in snapshot_contents.items():
        (tmp_path / filename).write_text(content)
    snapshot_hashes = {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in snapshot_contents
    }
    paired_values = {
        "schema_version": "1",
        "build_schema": "fepp-input-build-v5",
        "suite_release": "Suite TEST",
        "allow_microstate_mismatch": "0",
        "open_receptor_sha256": "c" * 64,
        "closed_receptor_sha256": "d" * 64,
        "ligand_bundle_sha256": snapshot_hashes["input_ligands.maegz"],
        "ligand_validation_sha256": snapshot_hashes["input_ligand_validation.json"],
        "receptor_microstates_sha256": snapshot_hashes["input_receptor_microstates.json"],
        "open_map_sha256": snapshot_hashes["input_map.fmp"],
        "closed_map_sha256": "e" * 64,
        "open_edge_sha256": snapshot_hashes["input_map.edge"],
        "closed_edge_sha256": snapshot_hashes["input_map.edge"],
        "open_map_provenance_sha256": snapshot_hashes["input_map.provenance"],
        "closed_map_provenance_sha256": "f" * 64,
        "open_atom_mapping_fingerprint": mapping_fingerprint,
        "closed_atom_mapping_fingerprint": mapping_fingerprint,
    }
    paired_path = tmp_path / "input_paired_inputs.tsv"
    paired_path.write_text(
        "\n".join(f"{key}\t{value}" for key, value in paired_values.items()) + "\n"
    )
    snapshot_hashes[paired_path.name] = hashlib.sha256(paired_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\n".join([
            "schema_version\t1",
            "state\topen",
            "seed\t2041",
            "jobname\tcompleted",
            "prepare_only\t0",
            "suite_release\tSuite TEST",
            "forcefield\tOPLS4",
            "custom_charge_mode\tassign",
            "water\tSPC",
            "ensemble\tmuVT",
            "time_ps\t5000",
            "equilibration_time_ps\t20",
            "lambda_windows\t12",
            "salt_molar\t0.0",
            "input_his267_state\tHIE",
            "input_prep_ph\t7.0",
            "input_prep_rmsd_A\t0.3",
            "input_map_topology\tnormal",
            "input_allow_microstate_mismatch\t0",
            f"map_sha256\t{snapshot_hashes['input_map.fmp']}",
            f"edge_sha256\t{snapshot_hashes['input_map.edge']}",
            f"map_provenance_sha256\t{snapshot_hashes['input_map.provenance']}",
            f"atom_mapping_fingerprint\t{mapping_fingerprint}",
            f"atom_mapping_json_sha256\t{snapshot_hashes['input_map_mappings.json']}",
            f"ligand_bundle_sha256\t{snapshot_hashes['input_ligands.maegz']}",
            f"ligand_validation_sha256\t{snapshot_hashes['input_ligand_validation.json']}",
            f"receptor_microstates_sha256\t{snapshot_hashes['input_receptor_microstates.json']}",
            f"paired_inputs_sha256\t{snapshot_hashes['input_paired_inputs.tsv']}",
        ])
        + "\n"
    )
    output = tmp_path / "open.csv"
    command = [
        sys.executable,
        str(FEPP_DIR / "extract_fep_results.py"),
        str(fmp),
        "--state",
        "open",
        "--schrodinger",
        str(suite),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    parsed = analysis.read_edge_file(output, expected_state="open")
    assert len(parsed.edges) == 1
    assert parsed.edges[0].source == "A"
    assert parsed.edges[0].target == "B"
    assert parsed.edges[0].ddg == pytest.approx(1.25)
    assert parsed.schema == "normalized_v2"
    assert parsed.run_seed == 2041
    assert parsed.run_manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    provenance_path = output.with_suffix(".csv.provenance.json")
    assert provenance_path.is_file()
    provenance = json.loads(provenance_path.read_text())
    assert provenance["schema_version"] == 3
    assert provenance["publication_status"] == "complete_commit_marker"
    assert provenance["input_environment_fingerprint"] == "c" * 64
    assert provenance["completed_environment_fingerprint"] == "c" * 64
    assert output.with_suffix(".csv.lock").is_file()
    assert output.with_name("open.vendor_edges.csv").is_file()

    stale_nodes = output.with_name("open.vendor_nodes.csv")
    stale_nodes.write_text("stale previous export\n")
    repeated = subprocess.run(command, text=True, capture_output=True, check=False)
    assert repeated.returncode == 0, repeated.stderr
    assert not stale_nodes.exists()

    blank_output = tmp_path / "blank.csv"
    blank_env = dict(os.environ, FAKE_BLANK="1")
    blank = subprocess.run(
        [*command[:-1], str(blank_output)],
        text=True,
        capture_output=True,
        check=False,
        env=blank_env,
    )
    assert blank.returncode != 0
    assert "blank required field 'bennett_ddg'" in blank.stderr
    assert not blank_output.exists()

    wrong_output = tmp_path / "wrong.csv"
    wrong_env = dict(os.environ, FAKE_WRONG_EDGE="1")
    wrong = subprocess.run(
        [*command[:-1], str(wrong_output)],
        text=True,
        capture_output=True,
        check=False,
        env=wrong_env,
    )
    assert wrong.returncode != 0
    assert "completed FMP edge set does not match" in wrong.stderr
    assert not wrong_output.exists()

    wrong_mapping_output = tmp_path / "wrong_mapping.csv"
    wrong_mapping_env = dict(os.environ, FAKE_WRONG_MAPPING="1")
    wrong_mapping = subprocess.run(
        [*command[:-1], str(wrong_mapping_output)],
        text=True,
        capture_output=True,
        check=False,
        env=wrong_mapping_env,
    )
    assert wrong_mapping.returncode != 0
    assert "completed FMP atom mapping does not match" in wrong_mapping.stderr
    assert not wrong_mapping_output.exists()

    wrong_environment_output = tmp_path / "wrong_environment.csv"
    wrong_environment_env = dict(os.environ, FAKE_WRONG_ENVIRONMENT="1")
    wrong_environment = subprocess.run(
        [*command[:-1], str(wrong_environment_output)],
        text=True,
        capture_output=True,
        check=False,
        env=wrong_environment_env,
    )
    assert wrong_environment.returncode != 0
    assert "completed FMP receptor/environment does not match" in wrong_environment.stderr
    assert not wrong_environment_output.exists()

    (suite / "version.txt").unlink()
    missing_release_output = tmp_path / "missing_release.csv"
    missing_release = subprocess.run(
        [*command[:-1], str(missing_release_output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_release.returncode != 0
    assert "Suite release metadata not found" in missing_release.stderr
    assert not missing_release_output.exists()
