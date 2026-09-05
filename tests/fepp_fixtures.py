"""Shared committed FEP+ edge-result factories for scientific and I/O tests."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from mdpp.core import fepp_results as reader

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


def _write_edges(
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
        *reader.INPUT_LINEAGE_FIELDS,
        "vendor_edge_table_sha256",
        *reader.QC_FIELDS,
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
                    else int(identity[:8], 16) % (reader.MAX_SEED + 1)
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
                **dict.fromkeys(reader.QC_FIELDS, qc_rating),
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
    input_lineage = {field: first_row[field] for field in reader.INPUT_LINEAGE_FIELDS}
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


SINGLE_EDGE_HASH = "1" * 64
SINGLE_MAPPING_FINGERPRINT = "2" * 64
SINGLE_LIGAND_BUNDLE_HASH = "3" * 64
SINGLE_LIGAND_VALIDATION_HASH = "4" * 64
SINGLE_STATE_INPUTS_HASH = "5" * 64
SINGLE_ENVIRONMENT_FINGERPRINT = "6" * 64
SINGLE_TEST_PROTOCOL = {
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
    "edge_sha256": SINGLE_EDGE_HASH,
    "atom_mapping_fingerprint": SINGLE_MAPPING_FINGERPRINT,
    "ligand_bundle_sha256": SINGLE_LIGAND_BUNDLE_HASH,
    "ligand_validation_sha256": SINGLE_LIGAND_VALIDATION_HASH,
    "workflow_type": "single_rbfe",
    "state_inputs_sha256": SINGLE_STATE_INPUTS_HASH,
}


def _write_single_edges(
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
    protocol_fields = dict(SINGLE_TEST_PROTOCOL)
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
        *reader.SINGLE_INPUT_LINEAGE_FIELDS,
        "vendor_edge_table_sha256",
        *reader.QC_FIELDS,
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
        "run_seed": int(identity[:8], 16) % (reader.MAX_SEED + 1),
        "source_fmp_sha256": hashlib.sha256(f"source:{path}".encode()).hexdigest(),
        "input_map_sha256": input_map_sha256 or hashlib.sha256(f"map:{state}".encode()).hexdigest(),
        "input_edge_sha256": SINGLE_EDGE_HASH,
        "input_map_provenance_sha256": hashlib.sha256(
            f"map-provenance:{state}".encode()
        ).hexdigest(),
        "input_atom_mapping_fingerprint": SINGLE_MAPPING_FINGERPRINT,
        "input_atom_mapping_json_sha256": hashlib.sha256(
            f"mapping-json:{state}".encode()
        ).hexdigest(),
        "input_ligand_bundle_sha256": SINGLE_LIGAND_BUNDLE_HASH,
        "input_ligand_validation_sha256": SINGLE_LIGAND_VALIDATION_HASH,
        "input_state_inputs_sha256": SINGLE_STATE_INPUTS_HASH,
        "vendor_edge_table_sha256": hashlib.sha256(f"vendor:{path}".encode()).hexdigest(),
        **dict.fromkeys(reader.QC_FIELDS, "Good"),
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
        field: str(row_identity[field]) for field in reader.SINGLE_INPUT_LINEAGE_FIELDS
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
        "completed_atom_mapping_fingerprint": SINGLE_MAPPING_FINGERPRINT,
        "input_environment_fingerprint": SINGLE_ENVIRONMENT_FINGERPRINT,
        "completed_environment_fingerprint": SINGLE_ENVIRONMENT_FINGERPRINT,
    }
    path.with_suffix(f"{path.suffix}.lock").touch()
    path.with_suffix(f"{path.suffix}.provenance.json").write_text(
        json.dumps(publication, indent=2) + "\n"
    )


@pytest.fixture()
def write_edges() -> Callable[..., None]:
    """Return a factory for committed paired-state schema-v2 result CSVs."""
    return _write_edges


@pytest.fixture()
def write_single_edges() -> Callable[..., None]:
    """Return a factory for committed single-state schema-v3 result CSVs."""
    return _write_single_edges
