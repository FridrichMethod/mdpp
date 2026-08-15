"""Regression tests for one-conformation FEP+ RBFE analysis."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

FEPP_DIR = Path(__file__).parents[3] / "examples" / "fepp"
sys.path.insert(0, str(FEPP_DIR))


def load_script(name: str) -> ModuleType:
    """Load one FEPP example script as a test module."""
    path = FEPP_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_single_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


paired_analysis = load_script("analyze_fep_results")
single_analysis = load_script("analyze_rbfe_results")
extractor = load_script("extract_fep_results")

EDGE_HASH = "1" * 64
MAPPING_FINGERPRINT = "2" * 64
LIGAND_BUNDLE_HASH = "3" * 64
LIGAND_VALIDATION_HASH = "4" * 64
STATE_INPUTS_HASH = "5" * 64
ENVIRONMENT_FINGERPRINT = "6" * 64
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
    "edge_sha256": EDGE_HASH,
    "atom_mapping_fingerprint": MAPPING_FINGERPRINT,
    "ligand_bundle_sha256": LIGAND_BUNDLE_HASH,
    "ligand_validation_sha256": LIGAND_VALIDATION_HASH,
    "workflow_type": "single_rbfe",
    "state_inputs_sha256": STATE_INPUTS_HASH,
}


def write_single_edges(
    path: Path,
    state: str,
    rows: list[tuple[str, str, float, float]],
    *,
    unit: str = "kcal/mol",
    input_map_sha256: str | None = None,
    protocol: dict[str, str] | None = None,
) -> None:
    """Write one committed normalized-v3 result file for tests."""
    identity = hashlib.sha256(str(path).encode()).hexdigest()
    protocol_fields = dict(TEST_PROTOCOL)
    if protocol is not None:
        protocol_fields.update(protocol)
    protocol_json = json.dumps(protocol_fields, sort_keys=True, separators=(",", ":"))
    fieldnames = [
        "schema_version",
        "engine",
        "workflow_type",
        "protocol_fingerprint",
        "protocol_json",
        "run_manifest_sha256",
        "run_id",
        "run_seed",
        "source_fmp_sha256",
        *paired_analysis.SINGLE_INPUT_LINEAGE_FIELDS,
        "vendor_edge_table_sha256",
        *paired_analysis.QC_FIELDS,
        "state",
        "edge_id",
        "source",
        "target",
        "ddg",
        "standard_uncertainty",
        "unit",
        "sign_convention",
    ]
    row_identity = {
        "schema_version": 3,
        "engine": "fep_plus",
        "workflow_type": "single_rbfe",
        "protocol_fingerprint": hashlib.sha256(protocol_json.encode()).hexdigest(),
        "protocol_json": protocol_json,
        "run_manifest_sha256": identity,
        "run_id": path.stem,
        "run_seed": int(identity[:8], 16) % (paired_analysis.MAX_SEED + 1),
        "source_fmp_sha256": hashlib.sha256(f"source:{path}".encode()).hexdigest(),
        "input_map_sha256": input_map_sha256 or hashlib.sha256(f"map:{state}".encode()).hexdigest(),
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
        "input_state_inputs_sha256": STATE_INPUTS_HASH,
        "vendor_edge_table_sha256": hashlib.sha256(f"vendor:{path}".encode()).hexdigest(),
        **dict.fromkeys(paired_analysis.QC_FIELDS, "Good"),
        "state": state,
        "unit": unit,
        "sign_convention": "target_minus_source",
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (source, target, ddg, uncertainty) in enumerate(rows):
            writer.writerow({
                **row_identity,
                "edge_id": f"e{index:03d}",
                "source": source,
                "target": target,
                "ddg": ddg,
                "standard_uncertainty": uncertainty,
            })
    input_lineage = {
        field: str(row_identity[field]) for field in paired_analysis.SINGLE_INPUT_LINEAGE_FIELDS
    }
    publication = {
        "schema_version": 3,
        "publication_status": "complete_commit_marker",
        "normalized_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_manifest_sha256": row_identity["run_manifest_sha256"],
        "run_id": row_identity["run_id"],
        "run_seed": row_identity["run_seed"],
        "source_fmp_sha256": row_identity["source_fmp_sha256"],
        "state": state,
        "workflow_type": "single_rbfe",
        "normalized_schema_version": 3,
        "protocol_fingerprint": row_identity["protocol_fingerprint"],
        "scientific_protocol": protocol_fields,
        "input_lineage": input_lineage,
        "completed_atom_mapping_fingerprint": MAPPING_FINGERPRINT,
        "input_environment_fingerprint": ENVIRONMENT_FINGERPRINT,
        "completed_environment_fingerprint": ENVIRONMENT_FINGERPRINT,
    }
    path.with_suffix(f"{path.suffix}.lock").touch()
    path.with_suffix(f"{path.suffix}.provenance.json").write_text(
        json.dumps(publication, indent=2) + "\n"
    )


def test_single_state_tree_covariance_pairwise_and_output_bundle(tmp_path: Path) -> None:
    result_csv = tmp_path / "open.csv"
    write_single_edges(
        result_csv,
        "open",
        [("A", "B", 1.0, 0.2), ("B", "C", 2.0, 0.3)],
    )

    result, tables = single_analysis.analyze([result_csv], state="open", reference="A")
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
    single_analysis.write_outputs(result, tables, output_dir=output_dir)
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


def test_pairwise_contrasts_are_reference_invariant_and_units_convert(tmp_path: Path) -> None:
    result_csv = tmp_path / "open.csv"
    write_single_edges(
        result_csv,
        "open",
        [("A", "B", 1.0, 0.2), ("B", "C", 2.0, 0.3)],
    )
    _, tables_a = single_analysis.analyze([result_csv], state="open", reference="A")
    result_b, tables_b = single_analysis.analyze(
        [result_csv], state="open", reference="B", output_unit="kJ/mol"
    )
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
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("Ligand1,Ligand2,bennett_ddg,bennett_ddg_error\nA,B,1.0,0.2\n")
    with pytest.raises(ValueError, match="requires normalized exports"):
        single_analysis.analyze([raw], state="open", reference="A")

    repeat_a = tmp_path / "repeat_a.csv"
    repeat_b = tmp_path / "repeat_b.csv"
    rows = [("A", "B", 1.0, 0.2)]
    write_single_edges(repeat_a, "open", rows)
    write_single_edges(repeat_b, "open", rows, input_map_sha256="f" * 64)
    with pytest.raises(ValueError, match="repeat input lineage differs"):
        single_analysis.analyze([repeat_a, repeat_b], state="open", reference="A")


def test_normalized_v3_rejects_legacy_commit_marker(tmp_path: Path) -> None:
    result_csv = tmp_path / "open.csv"
    write_single_edges(result_csv, "open", [("A", "B", 1.0, 0.2)])
    marker_path = result_csv.with_suffix(".csv.provenance.json")
    marker = json.loads(marker_path.read_text())
    marker["schema_version"] = 2
    marker.pop("input_environment_fingerprint")
    marker.pop("completed_environment_fingerprint")
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")

    with pytest.raises(ValueError, match="requires commit-marker schema 3"):
        paired_analysis.read_edge_file(result_csv, expected_state="open")

    write_single_edges(result_csv, "open", [("A", "B", 1.0, 0.2)])
    marker = json.loads(marker_path.read_text())
    marker.pop("workflow_type")
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    with pytest.raises(ValueError, match="schema 3 commit marker missing fields"):
        paired_analysis.read_edge_file(result_csv, expected_state="open")


def test_paired_analyzer_rejects_standalone_exports(tmp_path: Path) -> None:
    open_csv = tmp_path / "open.csv"
    closed_csv = tmp_path / "closed.csv"
    rows = [("A", "B", 1.0, 0.2)]
    write_single_edges(open_csv, "open", rows)
    write_single_edges(closed_csv, "closed", rows)
    with pytest.raises(ValueError, match="paired-state analysis requires normalized exports"):
        paired_analysis.analyze([open_csv], [closed_csv], reference="A")


def test_extractor_validates_standalone_manifest_and_state_cohort(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "version.txt").write_text("Suite TEST\n")
    runner = suite / "run"
    runner.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        output="$2"
        shift 2
    else
        shift
    fi
done
if [[ "${output}" == *mappings.json ]]; then
    printf '{"schema_version":2,"mapping_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","environment_fingerprint":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","edges":[]}\n' >"${output}"
else
    printf 'Ligand1,Ligand2,bennett_ddg,bennett_ddg_error,convergence,ligand RMSD,REST exchange,CCC Conv\nB,A,-1.25,0.2,Good,Good,Good,Good\n' >"${output}_ddG.csv"
fi
"""
    )
    runner.chmod(0o755)
    fmp = tmp_path / "standalone_out.fmp"
    fmp.write_text("completed map\n")
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
        "input_ligands.maegz": "ligands\n",
        "input_ligand_validation.json": '{"valid":true}\n',
    }
    for filename, content in snapshot_contents.items():
        (tmp_path / filename).write_text(content)
    hashes = {
        filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        for filename in snapshot_contents
    }
    state_values = {
        "schema_version": "1",
        "build_schema": "fepp-input-build-v5",
        "suite_release": "Suite TEST",
        "state": "open",
        "receptor_sha256": "b" * 64,
        "receptor_provenance_sha256": "c" * 64,
        "pose_viewer_sha256": "d" * 64,
        "pose_viewer_provenance_sha256": "e" * 64,
        "ligand_bundle_sha256": hashes["input_ligands.maegz"],
        "ligand_validation_sha256": hashes["input_ligand_validation.json"],
        "map_sha256": hashes["input_map.fmp"],
        "edge_sha256": hashes["input_map.edge"],
        "map_provenance_sha256": hashes["input_map.provenance"],
        "atom_mapping_fingerprint": mapping_fingerprint,
        "atom_mapping_json_sha256": hashes["input_map_mappings.json"],
    }
    state_path = tmp_path / "input_state_inputs.tsv"
    state_path.write_text(
        "\n".join(f"{key}\t{value}" for key, value in state_values.items()) + "\n"
    )
    state_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_values = {
        "schema_version": "2",
        "workflow_type": "single_rbfe",
        "state": "open",
        "seed": "2041",
        "jobname": "standalone",
        "prepare_only": "0",
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
        "map_sha256": hashes["input_map.fmp"],
        "edge_sha256": hashes["input_map.edge"],
        "map_provenance_sha256": hashes["input_map.provenance"],
        "atom_mapping_fingerprint": mapping_fingerprint,
        "atom_mapping_json_sha256": hashes["input_map_mappings.json"],
        "ligand_bundle_sha256": hashes["input_ligands.maegz"],
        "ligand_validation_sha256": hashes["input_ligand_validation.json"],
        "state_inputs_sha256": state_hash,
    }
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\n".join(f"{key}\t{value}" for key, value in manifest_values.items()) + "\n"
    )

    run = extractor._read_manifest(
        manifest,
        state="open",
        fmp=fmp,
        schrodinger=suite,
    )
    assert run.workflow_type == "single_rbfe"
    assert run.normalized_schema_version == 3
    assert run.input_lineage["input_state_inputs_sha256"] == state_hash
    assert run.expected_edge_pairs == frozenset({("A", "B")})

    output = tmp_path / "standalone.csv"
    assert (
        extractor.extract(
            fmp,
            output,
            state="open",
            schrodinger=suite,
            manifest=manifest,
        )
        == 1
    )
    parsed = paired_analysis.read_edge_file(output, expected_state="open")
    assert parsed.schema == "normalized_v3"
    assert parsed.workflow_type == "single_rbfe"
    assert parsed.input_lineage is not None
    assert parsed.input_lineage["input_state_inputs_sha256"] == state_hash

    state_values["state"] = "closed"
    state_path.write_text(
        "\n".join(f"{key}\t{value}" for key, value in state_values.items()) + "\n"
    )
    manifest_values["state_inputs_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest.write_text(
        "\n".join(f"{key}\t{value}" for key, value in manifest_values.items()) + "\n"
    )
    with pytest.raises(ValueError, match="state does not match"):
        extractor._read_manifest(
            manifest,
            state="open",
            fmp=fmp,
            schrodinger=suite,
        )
