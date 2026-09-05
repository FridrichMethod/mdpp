"""Validated FEP+ and OpenFE edge-result CSV reading and publication provenance.

Edge observations use kcal/mol internally, independent of the input CSV unit.
Normalized FEP+ inputs retain the existing schema-v2/v3 commit-marker contract.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from mdpp._types import StrPath

COMMON_REQUIRED_PROTOCOL_FIELDS = {
    "suite_release",
    "forcefield",
    "custom_charge_mode",
    "water",
    "ensemble",
    "time_ps",
    "equilibration_time_ps",
    "lambda_windows",
    "salt_molar",
    "input_his267_state",
    "input_prep_ph",
    "input_prep_rmsd_A",
    "input_map_topology",
    "edge_sha256",
    "atom_mapping_fingerprint",
    "ligand_bundle_sha256",
    "ligand_validation_sha256",
}
PAIRED_REQUIRED_PROTOCOL_FIELDS = {
    *COMMON_REQUIRED_PROTOCOL_FIELDS,
    "input_allow_microstate_mismatch",
    "receptor_microstates_sha256",
    "paired_inputs_sha256",
}
SINGLE_REQUIRED_PROTOCOL_FIELDS = {
    *COMMON_REQUIRED_PROTOCOL_FIELDS,
    "workflow_type",
    "state_inputs_sha256",
}
REQUIRED_PROTOCOL_FIELDS = PAIRED_REQUIRED_PROTOCOL_FIELDS
COMMON_INPUT_LINEAGE_FIELDS = (
    "input_map_sha256",
    "input_edge_sha256",
    "input_map_provenance_sha256",
    "input_atom_mapping_fingerprint",
    "input_atom_mapping_json_sha256",
    "input_ligand_bundle_sha256",
    "input_ligand_validation_sha256",
)
INPUT_LINEAGE_FIELDS = (
    *COMMON_INPUT_LINEAGE_FIELDS,
    "input_receptor_microstates_sha256",
    "input_paired_inputs_sha256",
)
SINGLE_INPUT_LINEAGE_FIELDS = (
    *COMMON_INPUT_LINEAGE_FIELDS,
    "input_state_inputs_sha256",
)
COMMON_PROTOCOL_TO_LINEAGE = {
    "edge_sha256": "input_edge_sha256",
    "atom_mapping_fingerprint": "input_atom_mapping_fingerprint",
    "ligand_bundle_sha256": "input_ligand_bundle_sha256",
    "ligand_validation_sha256": "input_ligand_validation_sha256",
}
PROTOCOL_TO_LINEAGE = {
    **COMMON_PROTOCOL_TO_LINEAGE,
    "receptor_microstates_sha256": "input_receptor_microstates_sha256",
    "paired_inputs_sha256": "input_paired_inputs_sha256",
}
SINGLE_PROTOCOL_TO_LINEAGE = {
    **COMMON_PROTOCOL_TO_LINEAGE,
    "state_inputs_sha256": "input_state_inputs_sha256",
}
QC_FIELDS = (
    "qc_convergence",
    "qc_ligand_rmsd",
    "qc_rest_exchange",
    "qc_ccc_convergence",
)
QC_RATINGS = {"Good", "Fair", "Bad", "N/A"}
COMMON_NORMALIZED_FIELDS = {
    "schema_version",
    "engine",
    "protocol_fingerprint",
    "protocol_json",
    "run_manifest_sha256",
    "run_id",
    "run_seed",
    "source_fmp_sha256",
    "vendor_edge_table_sha256",
    *QC_FIELDS,
    "state",
    "edge_id",
    "source",
    "target",
    "ddg",
    "standard_uncertainty",
    "unit",
    "sign_convention",
}
NORMALIZED_V2_FIELDS = COMMON_NORMALIZED_FIELDS | set(INPUT_LINEAGE_FIELDS)
NORMALIZED_V3_FIELDS = (
    COMMON_NORMALIZED_FIELDS | {"workflow_type"} | set(SINGLE_INPUT_LINEAGE_FIELDS)
)
NORMALIZED_FIELDS = NORMALIZED_V2_FIELDS
FEP_PLUS_FIELDS = {"Ligand1", "Ligand2", "bennett_ddg", "bennett_ddg_error"}
OPENFE_FIELDS = {
    "ligand_i",
    "ligand_j",
    "DDG(i->j) (kcal/mol)",
    "uncertainty (kcal/mol)",
}
SUPPORTED_UNITS = {"kcal/mol": 1.0, "kJ/mol": 1.0 / 4.184}
ANCHOR_QUANTITIES = {
    "binding_free_energy_open_minus_closed",
    "holo_conformational_free_energy_open_minus_closed",
}
MAX_SEED = 2147483647


@dataclass(frozen=True, slots=True)
class Edge:
    """One directed relative binding free-energy observation in kcal/mol."""

    source: str
    target: str
    ddg: float
    uncertainty: float
    edge_id: str


@dataclass(frozen=True, slots=True)
class ParsedEdgeFile:
    """Validated observations and provenance from one result CSV."""

    path: Path
    schema: str
    state: str
    engine: str
    protocol_fingerprint: str | None
    protocol_json: str | None
    run_manifest_sha256: str | None
    run_id: str | None
    run_seed: int | None
    source_fmp_sha256: str | None
    input_lineage: dict[str, str] | None
    vendor_edge_table_sha256: str | None
    quality_control: dict[str, Any] | None
    edges: tuple[Edge, ...]
    sha256: str
    workflow_type: str | None = None


def _finite_float(value: str | None, *, field: str, path: Path, row_number: int) -> float:
    """Parse one required finite floating-point CSV field."""
    if value is None or not value.strip():
        raise ValueError(f"{path}: row {row_number}: blank required field {field!r}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}: row {row_number}: {field!r} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}: row {row_number}: {field!r} must be finite")
    return parsed


def _required_text(value: str | None, *, field: str, path: Path, row_number: int) -> str:
    """Return a stripped, nonempty CSV field."""
    if value is None or not value.strip():
        raise ValueError(f"{path}: row {row_number}: blank required field {field!r}")
    return value.strip()


def _required_sha256(value: str | None, *, field: str, path: Path, row_number: int) -> str:
    """Return one required lowercase SHA-256 field."""
    digest = _required_text(value, field=field, path=path, row_number=row_number)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{path}: row {row_number}: {field} must be lowercase SHA-256")
    return digest


def _validate_protocol_values(protocol: dict[str, str], *, path: Path, row_number: int) -> None:
    """Validate launcher-controlled categorical values and numeric ranges."""
    allowed = {
        "forcefield": {"OPLS4", "OPLS5"},
        "custom_charge_mode": {"assign"},
        "water": {
            "SPC",
            "SPCE",
            "TIP3P",
            "TIP3P_CHARMM",
            "TIP4P",
            "TIP4PEW",
            "TIP4P2005",
            "TIP5P",
            "TIP4PD",
        },
        "ensemble": {"muVT", "NPT", "NVT"},
        "input_his267_state": {"HID", "HIE", "HIP", "auto"},
        "input_map_topology": {"full", "normal", "star", "windmill"},
    }
    if "input_allow_microstate_mismatch" in protocol:
        allowed["input_allow_microstate_mismatch"] = {"0", "1"}
    if "workflow_type" in protocol:
        allowed["workflow_type"] = {"single_rbfe"}
    for field, choices in allowed.items():
        if protocol[field] not in choices:
            raise ValueError(
                f"{path}: row {row_number}: unsupported protocol {field}={protocol[field]!r}"
            )
    numeric_bounds = {
        "time_ps": (500.0, None),
        "equilibration_time_ps": (0.0, None),
        "salt_molar": (0.0, None),
        "input_prep_ph": (0.0, 14.0),
        "input_prep_rmsd_A": (0.0, None),
    }
    for field, (lower, upper) in numeric_bounds.items():
        try:
            value = float(protocol[field])
        except ValueError as exc:
            raise ValueError(f"{path}: row {row_number}: protocol {field} must be numeric") from exc
        lower_ok = value >= lower if field != "input_prep_rmsd_A" else value > lower
        if not math.isfinite(value) or not lower_ok or (upper is not None and value > upper):
            raise ValueError(f"{path}: row {row_number}: protocol {field} is out of range")
    try:
        windows = int(protocol["lambda_windows"])
    except ValueError as exc:
        raise ValueError(f"{path}: row {row_number}: lambda_windows must be an integer") from exc
    if windows < 2:
        raise ValueError(f"{path}: row {row_number}: lambda_windows must be >=2")
    hash_fields = set(COMMON_PROTOCOL_TO_LINEAGE)
    hash_fields.update(
        field
        for field in (
            "receptor_microstates_sha256",
            "paired_inputs_sha256",
            "state_inputs_sha256",
        )
        if field in protocol
    )
    for field in hash_fields:
        _required_sha256(protocol[field], field=field, path=path, row_number=row_number)


def _required_protocol_json(
    value: str | None,
    *,
    path: Path,
    row_number: int,
    required_fields: set[str],
) -> str:
    """Validate and canonicalize the scientific protocol JSON."""
    text = _required_text(value, field="protocol_json", path=path, row_number=row_number)
    try:
        protocol = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: row {row_number}: protocol_json is not valid JSON") from exc
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError(f"{path}: row {row_number}: protocol_json must be a nonempty object")
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in protocol.items()):
        raise ValueError(f"{path}: row {row_number}: protocol_json keys and values must be strings")
    missing = sorted(required_fields - protocol.keys())
    if missing:
        raise ValueError(f"{path}: row {row_number}: protocol_json is missing fields: {missing}")
    _validate_protocol_values(protocol, path=path, row_number=row_number)
    return json.dumps(protocol, sort_keys=True, separators=(",", ":"))


def _canonical_edge(source: str, target: str, ddg: float) -> tuple[str, str, float]:
    """Orient an edge lexicographically without changing its physical meaning."""
    if source == target:
        raise ValueError(f"self-edge is not valid: {source!r} -> {target!r}")
    if source < target:
        return source, target, ddg
    return target, source, -ddg


def _summarize_qc(records: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize categorical vendor QC while retaining warning edges."""
    counts = {
        field: {rating: sum(record[field] == rating for record in records) for rating in QC_RATINGS}
        for field in QC_FIELDS
    }
    warning_edges = [
        record for record in records if any(record[field] != "Good" for field in QC_FIELDS)
    ]
    return {
        "edge_count": len(records),
        "all_ratings_good": not warning_edges,
        "warning_edge_count": len(warning_edges),
        "rating_counts": counts,
        "warning_edges": warning_edges,
        "policy": "diagnostic_only_no_automatic_edge_exclusion",
    }


def _detect_schema(fieldnames: set[str], path: Path) -> str:
    """Identify one supported result CSV schema."""
    if fieldnames >= NORMALIZED_V3_FIELDS:
        return "normalized_v3"
    if fieldnames >= NORMALIZED_V2_FIELDS:
        return "normalized_v2"
    if fieldnames >= FEP_PLUS_FIELDS:
        return "fep_plus_fmp2excel"
    if fieldnames >= OPENFE_FIELDS:
        return "openfe_gather"
    raise ValueError(
        f"{path}: unsupported CSV columns; expected normalized fields "
        f"{sorted(NORMALIZED_V2_FIELDS)} or {sorted(NORMALIZED_V3_FIELDS)} "
        f"or FEP+ fields {sorted(FEP_PLUS_FIELDS)}"
    )


def _parse_csv_snapshot(
    raw_bytes: bytes,
    *,
    path: Path,
) -> tuple[str, list[dict[str, str]]]:
    """Parse one immutable byte snapshot and return its schema and rows."""
    try:
        text = raw_bytes.decode()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: result CSV is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise ValueError(f"{path}: missing CSV header")
    schema = _detect_schema(set(reader.fieldnames), path)
    return schema, list(reader)


def _read_result_snapshot(
    path: Path,
    *,
    require_commit_marker: bool,
) -> tuple[bytes, str, list[dict[str, str]], dict[str, Any] | None]:
    """Read a CSV and its commit marker under the publisher's shared lock."""
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    provenance_path = path.with_suffix(f"{path.suffix}.provenance.json")

    def read_locked() -> tuple[bytes, str, list[dict[str, str]], dict[str, Any] | None]:
        raw_bytes = path.read_bytes()
        schema, rows = _parse_csv_snapshot(raw_bytes, path=path)
        if not schema.startswith("normalized_v") or not require_commit_marker:
            return raw_bytes, schema, rows, None
        if not provenance_path.is_file():
            raise ValueError(
                f"{path}: normalized result is incomplete; missing commit marker {provenance_path}"
            )
        try:
            publication = json.loads(provenance_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{provenance_path}: invalid commit-marker JSON") from exc
        if not isinstance(publication, dict):
            raise ValueError(f"{provenance_path}: commit marker must be a JSON object")
        if publication.get("schema_version") not in {2, 3}:
            raise ValueError(f"{provenance_path}: expected schema_version 2 or 3")
        if publication.get("publication_status") != "complete_commit_marker":
            raise ValueError(f"{provenance_path}: result publication is not complete")
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if publication.get("normalized_result_sha256") != digest:
            raise ValueError(f"{provenance_path}: normalized result SHA-256 does not match {path}")
        return raw_bytes, schema, rows, publication

    if lock_path.is_file():
        with lock_path.open("r") as lock_handle:
            # File publication uses POSIX locks; importing mdpp does not.
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            return read_locked()

    raw_bytes, schema, rows, publication = read_locked()
    if schema.startswith("normalized_v") and require_commit_marker:
        raise ValueError(f"{path}: normalized result is incomplete; missing lock {lock_path}")
    return raw_bytes, schema, rows, publication


def _validate_environment_fingerprints(
    path: Path,
    publication: dict[str, Any],
    *,
    required: bool,
) -> None:
    """Validate optional input/completed FMP environment fingerprints."""
    input_environment = publication.get("input_environment_fingerprint")
    completed_environment = publication.get("completed_environment_fingerprint")
    if input_environment is None and completed_environment is None:
        if required:
            raise ValueError(f"{path}: schema 3 commit marker requires environment fingerprints")
        return
    for field, fingerprint in (
        ("input_environment_fingerprint", input_environment),
        ("completed_environment_fingerprint", completed_environment),
    ):
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError(f"{path}: commit-marker {field} must be a lowercase SHA-256")
    if completed_environment != input_environment:
        raise ValueError(f"{path}: completed-FMP environment does not match input map")


def _validate_commit_marker(parsed: ParsedEdgeFile, publication: dict[str, Any]) -> None:
    """Bind normalized row metadata to the extractor's completion marker."""
    expected_scalars = {
        "run_manifest_sha256": parsed.run_manifest_sha256,
        "run_id": parsed.run_id,
        "run_seed": parsed.run_seed,
        "source_fmp_sha256": parsed.source_fmp_sha256,
        "state": parsed.state,
        "protocol_fingerprint": parsed.protocol_fingerprint,
    }
    for field, expected in expected_scalars.items():
        if publication.get(field) != expected:
            raise ValueError(f"{parsed.path}: commit-marker {field} does not match normalized rows")
    marker_schema_version = publication.get("schema_version")
    if marker_schema_version == 3:
        missing_marker_fields = sorted(
            field
            for field in ("workflow_type", "normalized_schema_version")
            if field not in publication
        )
        if missing_marker_fields:
            raise ValueError(
                f"{parsed.path}: schema 3 commit marker missing fields: {missing_marker_fields}"
            )
    if publication.get("workflow_type", parsed.workflow_type) != parsed.workflow_type:
        raise ValueError(f"{parsed.path}: commit-marker workflow_type does not match rows")
    if parsed.schema == "normalized_v3" and marker_schema_version != 3:
        raise ValueError(f"{parsed.path}: normalized-v3 result requires commit-marker schema 3")
    expected_schema_version = 3 if parsed.schema == "normalized_v3" else 2
    if publication.get("normalized_schema_version", expected_schema_version) != (
        expected_schema_version
    ):
        raise ValueError(
            f"{parsed.path}: commit-marker normalized_schema_version does not match rows"
        )
    if publication.get("input_lineage") != parsed.input_lineage:
        raise ValueError(f"{parsed.path}: commit-marker input lineage does not match rows")
    if parsed.protocol_json is None:
        raise ValueError(f"{parsed.path}: normalized rows are missing a scientific protocol")
    if publication.get("scientific_protocol") != json.loads(parsed.protocol_json):
        raise ValueError(f"{parsed.path}: commit-marker scientific protocol does not match rows")
    input_lineage = parsed.input_lineage or {}
    if publication.get("completed_atom_mapping_fingerprint") != input_lineage.get(
        "input_atom_mapping_fingerprint"
    ):
        raise ValueError(
            f"{parsed.path}: completed-FMP mapping fingerprint does not match input lineage"
        )
    _validate_environment_fingerprints(
        parsed.path,
        publication,
        required=publication.get("schema_version") == 3,
    )


def read_edge_file(  # noqa: C901
    path: StrPath,
    *,
    expected_state: str,
    require_commit_marker: bool = True,
) -> ParsedEdgeFile:
    """Read and strictly validate one edge-result CSV.

    Args:
        path: Normalized, FEP+ ``fmp2excel``, or OpenFE edge CSV.
        expected_state: Required state label (``open`` or ``closed``).
        require_commit_marker: Require extractor publication metadata for the
            normalized schema. Disable only for an unpublished staged file.

    Returns:
        Parsed edges converted to kcal/mol and canonical edge orientation.

    Raises:
        ValueError: If the schema, metadata, values, or graph edges are invalid.
    """
    path = Path(path)
    raw_bytes, schema, rows, publication = _read_result_snapshot(
        path,
        require_commit_marker=require_commit_marker,
    )
    if not rows:
        raise ValueError(f"{path}: no result rows (the FMP may not have completed)")

    edges: list[Edge] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    engines: set[str] = set()
    units: set[str] = set()
    protocol_fingerprints: set[str] = set()
    protocol_json_values: set[str] = set()
    run_manifest_hashes: set[str] = set()
    run_ids: set[str] = set()
    run_seeds: set[int] = set()
    source_fmp_hashes: set[str] = set()
    input_lineage_values: set[str] = set()
    vendor_edge_table_hashes: set[str] = set()
    workflow_types: set[str] = set()
    qc_records: list[dict[str, str]] = []
    for row_index, row in enumerate(rows, start=2):
        row_qc: dict[str, str] | None = None
        if schema in {"normalized_v2", "normalized_v3"}:
            schema_version = _required_text(
                row.get("schema_version"),
                field="schema_version",
                path=path,
                row_number=row_index,
            )
            expected_schema_version = "2" if schema == "normalized_v2" else "3"
            if schema_version != expected_schema_version:
                raise ValueError(
                    f"{path}: row {row_index}: unsupported schema_version {schema_version!r}"
                )
            lineage_fields: tuple[str, ...]
            if schema == "normalized_v2":
                workflow_type = "paired_open_closed"
                lineage_fields = INPUT_LINEAGE_FIELDS
                required_protocol_fields = PAIRED_REQUIRED_PROTOCOL_FIELDS
                protocol_to_lineage = PROTOCOL_TO_LINEAGE
            else:
                workflow_type = _required_text(
                    row.get("workflow_type"),
                    field="workflow_type",
                    path=path,
                    row_number=row_index,
                )
                if workflow_type != "single_rbfe":
                    raise ValueError(
                        f"{path}: row {row_index}: unsupported workflow_type {workflow_type!r}"
                    )
                lineage_fields = SINGLE_INPUT_LINEAGE_FIELDS
                required_protocol_fields = SINGLE_REQUIRED_PROTOCOL_FIELDS
                protocol_to_lineage = SINGLE_PROTOCOL_TO_LINEAGE
            state = _required_text(row.get("state"), field="state", path=path, row_number=row_index)
            if state != expected_state:
                raise ValueError(
                    f"{path}: row {row_index}: state {state!r} does not match "
                    f"expected {expected_state!r}"
                )
            sign = _required_text(
                row.get("sign_convention"),
                field="sign_convention",
                path=path,
                row_number=row_index,
            )
            if sign != "target_minus_source":
                raise ValueError(f"{path}: row {row_index}: unsupported sign convention {sign!r}")
            unit = _required_text(row.get("unit"), field="unit", path=path, row_number=row_index)
            if unit not in SUPPORTED_UNITS:
                raise ValueError(f"{path}: row {row_index}: unsupported unit {unit!r}")
            source = _required_text(
                row.get("source"), field="source", path=path, row_number=row_index
            )
            target = _required_text(
                row.get("target"), field="target", path=path, row_number=row_index
            )
            ddg = _finite_float(row.get("ddg"), field="ddg", path=path, row_number=row_index)
            uncertainty = _finite_float(
                row.get("standard_uncertainty"),
                field="standard_uncertainty",
                path=path,
                row_number=row_index,
            )
            factor = SUPPORTED_UNITS[unit]
            units.add(unit)
            ddg *= factor
            uncertainty *= factor
            edge_id = _required_text(
                row.get("edge_id"), field="edge_id", path=path, row_number=row_index
            )
            engine = _required_text(
                row.get("engine"), field="engine", path=path, row_number=row_index
            )
            protocol_fingerprint = _required_sha256(
                row.get("protocol_fingerprint"),
                field="protocol_fingerprint",
                path=path,
                row_number=row_index,
            )
            protocol_json = _required_protocol_json(
                row.get("protocol_json"),
                path=path,
                row_number=row_index,
                required_fields=required_protocol_fields,
            )
            computed_protocol_fingerprint = hashlib.sha256(protocol_json.encode()).hexdigest()
            if computed_protocol_fingerprint != protocol_fingerprint:
                raise ValueError(
                    f"{path}: row {row_index}: protocol_fingerprint does not match protocol_json"
                )
            run_manifest_sha256 = _required_sha256(
                row.get("run_manifest_sha256"),
                field="run_manifest_sha256",
                path=path,
                row_number=row_index,
            )
            run_id = _required_text(
                row.get("run_id"), field="run_id", path=path, row_number=row_index
            )
            seed_text = _required_text(
                row.get("run_seed"), field="run_seed", path=path, row_number=row_index
            )
            try:
                run_seed = int(seed_text)
            except ValueError as exc:
                raise ValueError(f"{path}: row {row_index}: run_seed must be an integer") from exc
            if not 0 <= run_seed <= MAX_SEED:
                raise ValueError(f"{path}: row {row_index}: run_seed must be in [0, {MAX_SEED}]")
            source_fmp_sha256 = _required_sha256(
                row.get("source_fmp_sha256"),
                field="source_fmp_sha256",
                path=path,
                row_number=row_index,
            )
            input_lineage = {
                field: _required_sha256(
                    row.get(field), field=field, path=path, row_number=row_index
                )
                for field in lineage_fields
            }
            vendor_edge_table_sha256 = _required_sha256(
                row.get("vendor_edge_table_sha256"),
                field="vendor_edge_table_sha256",
                path=path,
                row_number=row_index,
            )
            row_qc = {}
            for field in QC_FIELDS:
                rating = _required_text(
                    row.get(field), field=field, path=path, row_number=row_index
                )
                if rating not in QC_RATINGS:
                    raise ValueError(
                        f"{path}: row {row_index}: unsupported {field} rating {rating!r}"
                    )
                row_qc[field] = rating
            protocol = json.loads(protocol_json)
            for protocol_field, lineage_field in protocol_to_lineage.items():
                if protocol[protocol_field] != input_lineage[lineage_field]:
                    raise ValueError(
                        f"{path}: row {row_index}: protocol {protocol_field} does not match "
                        f"{lineage_field}"
                    )
            protocol_fingerprints.add(protocol_fingerprint)
            protocol_json_values.add(protocol_json)
            run_manifest_hashes.add(run_manifest_sha256)
            run_ids.add(run_id)
            run_seeds.add(run_seed)
            source_fmp_hashes.add(source_fmp_sha256)
            input_lineage_values.add(
                json.dumps(input_lineage, sort_keys=True, separators=(",", ":"))
            )
            vendor_edge_table_hashes.add(vendor_edge_table_sha256)
            workflow_types.add(workflow_type)
        elif schema == "fep_plus_fmp2excel":
            source = _required_text(
                row.get("Ligand1"), field="Ligand1", path=path, row_number=row_index
            )
            target = _required_text(
                row.get("Ligand2"), field="Ligand2", path=path, row_number=row_index
            )
            ddg = _finite_float(
                row.get("bennett_ddg"),
                field="bennett_ddg",
                path=path,
                row_number=row_index,
            )
            uncertainty = _finite_float(
                row.get("bennett_ddg_error"),
                field="bennett_ddg_error",
                path=path,
                row_number=row_index,
            )
            edge_id = f"e{row_index - 2:04d}"
            engine = "fep_plus"
        else:
            source = _required_text(
                row.get("ligand_i"), field="ligand_i", path=path, row_number=row_index
            )
            target = _required_text(
                row.get("ligand_j"), field="ligand_j", path=path, row_number=row_index
            )
            ddg = _finite_float(
                row.get("DDG(i->j) (kcal/mol)"),
                field="DDG(i->j) (kcal/mol)",
                path=path,
                row_number=row_index,
            )
            uncertainty = _finite_float(
                row.get("uncertainty (kcal/mol)"),
                field="uncertainty (kcal/mol)",
                path=path,
                row_number=row_index,
            )
            edge_id = f"e{row_index - 2:04d}"
            engine = "openfe"

        if uncertainty <= 0.0:
            raise ValueError(
                f"{path}: row {row_index}: standard uncertainty must be greater than zero"
            )
        canonical_source, canonical_target, canonical_ddg = _canonical_edge(source, target, ddg)
        pair = (canonical_source, canonical_target)
        if pair in seen_pairs:
            raise ValueError(f"{path}: duplicate unordered ligand pair {pair}")
        if edge_id in seen_ids:
            raise ValueError(f"{path}: duplicate edge_id {edge_id!r}")
        seen_pairs.add(pair)
        seen_ids.add(edge_id)
        engines.add(engine)
        if row_qc is not None:
            qc_records.append({"source": canonical_source, "target": canonical_target, **row_qc})
        edges.append(
            Edge(
                source=canonical_source,
                target=canonical_target,
                ddg=canonical_ddg,
                uncertainty=uncertainty,
                edge_id=edge_id,
            )
        )

    graph = nx.Graph((edge.source, edge.target) for edge in edges)
    if not nx.is_connected(graph):
        components = [sorted(component) for component in nx.connected_components(graph)]
        raise ValueError(f"{path}: result graph is disconnected: {components}")
    if len(engines) != 1:
        raise ValueError(f"{path}: normalized rows contain multiple engines: {sorted(engines)}")
    if len(units) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple units: {sorted(units)}")
    if len(protocol_fingerprints) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple protocol fingerprints")
    if len(protocol_json_values) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple scientific protocols")
    if len(run_manifest_hashes) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple run manifest hashes")
    if len(run_ids) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple run IDs")
    if len(run_seeds) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple run seeds")
    if len(source_fmp_hashes) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple source FMP hashes")
    if len(input_lineage_values) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple input lineages")
    if len(vendor_edge_table_hashes) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple vendor edge-table hashes")
    if len(workflow_types) > 1:
        raise ValueError(f"{path}: normalized rows contain multiple workflow types")
    input_lineage_json = next(iter(input_lineage_values), None)
    parsed = ParsedEdgeFile(
        path=path,
        schema=schema,
        state=expected_state,
        engine=engines.pop(),
        protocol_fingerprint=next(iter(protocol_fingerprints), None),
        protocol_json=next(iter(protocol_json_values), None),
        run_manifest_sha256=next(iter(run_manifest_hashes), None),
        run_id=next(iter(run_ids), None),
        run_seed=next(iter(run_seeds), None),
        source_fmp_sha256=next(iter(source_fmp_hashes), None),
        input_lineage=(json.loads(input_lineage_json) if input_lineage_json is not None else None),
        vendor_edge_table_sha256=next(iter(vendor_edge_table_hashes), None),
        quality_control=(_summarize_qc(qc_records) if qc_records else None),
        edges=tuple(sorted(edges, key=lambda edge: (edge.source, edge.target))),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        workflow_type=next(iter(workflow_types), None),
    )
    if publication is not None:
        _validate_commit_marker(parsed, publication)
    return parsed
