#!/usr/bin/env python3
"""Analyze paired open/closed FEP+ RBFE networks from raw Bennett edge results.

The two RBFE networks do not identify an absolute open-versus-closed binding
preference.  With reference ligand ``r``, this script estimates

    D_i^(r) = [G_open(i) - G_open(r)] - [G_closed(i) - G_closed(r)].

Negative values mean ligand ``i`` is more open-selective than the reference.
An optional, independently determined open-minus-closed anchor for the
reference converts these relative values into an absolute anchored quantity.

Paired analysis requires the provenance-bearing normalized schema written by
``extract_fep_results.py``.  Its low-level reader also accepts the raw
``*_ddG.csv`` written by Schrodinger's ``fmp2excel.py`` so that the extractor
can validate vendor output.  Only raw Bennett edge values and uncertainties
are fitted.  Vendor cycle-closure-adjusted values and ``ccc_ddg_error`` are
intentionally not statistical inputs.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.stats import chi2

REQUIRED_PROTOCOL_FIELDS = {
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
    "input_allow_microstate_mismatch",
    "edge_sha256",
    "atom_mapping_fingerprint",
    "ligand_bundle_sha256",
    "ligand_validation_sha256",
    "receptor_microstates_sha256",
    "paired_inputs_sha256",
}
INPUT_LINEAGE_FIELDS = (
    "input_map_sha256",
    "input_edge_sha256",
    "input_map_provenance_sha256",
    "input_atom_mapping_fingerprint",
    "input_atom_mapping_json_sha256",
    "input_ligand_bundle_sha256",
    "input_ligand_validation_sha256",
    "input_receptor_microstates_sha256",
    "input_paired_inputs_sha256",
)
PROTOCOL_TO_LINEAGE = {
    "edge_sha256": "input_edge_sha256",
    "atom_mapping_fingerprint": "input_atom_mapping_fingerprint",
    "ligand_bundle_sha256": "input_ligand_bundle_sha256",
    "ligand_validation_sha256": "input_ligand_validation_sha256",
    "receptor_microstates_sha256": "input_receptor_microstates_sha256",
    "paired_inputs_sha256": "input_paired_inputs_sha256",
}
QC_FIELDS = (
    "qc_convergence",
    "qc_ligand_rmsd",
    "qc_rest_exchange",
    "qc_ccc_convergence",
)
QC_RATINGS = {"Good", "Fair", "Bad", "N/A"}
NORMALIZED_FIELDS = {
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
} | set(INPUT_LINEAGE_FIELDS)
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


@dataclass(frozen=True, slots=True)
class CombinedEdge:
    """One edge combined across independent repeat result files."""

    source: str
    target: str
    ddg: float
    uncertainty: float
    repeat_count: int
    heterogeneity_chi2: float
    heterogeneity_dof: int
    fixed_effect_uncertainty: float | None = None
    heterogeneity_p_value: float | None = None
    uncertainty_scale_factor: float = 1.0


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
        "input_allow_microstate_mismatch": {"0", "1"},
    }
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
    for field in PROTOCOL_TO_LINEAGE:
        _required_sha256(protocol[field], field=field, path=path, row_number=row_number)


def _required_protocol_json(value: str | None, *, path: Path, row_number: int) -> str:
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
    missing = sorted(REQUIRED_PROTOCOL_FIELDS - protocol.keys())
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
    if fieldnames >= NORMALIZED_FIELDS:
        return "normalized_v2"
    if fieldnames >= FEP_PLUS_FIELDS:
        return "fep_plus_fmp2excel"
    if fieldnames >= OPENFE_FIELDS:
        return "openfe_gather"
    raise ValueError(
        f"{path}: unsupported CSV columns; expected normalized fields "
        f"{sorted(NORMALIZED_FIELDS)} or FEP+ fields {sorted(FEP_PLUS_FIELDS)}"
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
        if schema != "normalized_v2" or not require_commit_marker:
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
        if publication.get("schema_version") != 2:
            raise ValueError(f"{provenance_path}: expected schema_version 2")
        if publication.get("publication_status") != "complete_commit_marker":
            raise ValueError(f"{provenance_path}: result publication is not complete")
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if publication.get("normalized_result_sha256") != digest:
            raise ValueError(f"{provenance_path}: normalized result SHA-256 does not match {path}")
        return raw_bytes, schema, rows, publication

    if lock_path.is_file():
        with lock_path.open("r") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            return read_locked()

    raw_bytes, schema, rows, publication = read_locked()
    if schema == "normalized_v2" and require_commit_marker:
        raise ValueError(f"{path}: normalized result is incomplete; missing lock {lock_path}")
    return raw_bytes, schema, rows, publication


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


def read_edge_file(  # noqa: C901
    path: Path,
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
    qc_records: list[dict[str, str]] = []
    for row_index, row in enumerate(rows, start=2):
        row_qc: dict[str, str] | None = None
        if schema == "normalized_v2":
            schema_version = _required_text(
                row.get("schema_version"),
                field="schema_version",
                path=path,
                row_number=row_index,
            )
            if schema_version != "2":
                raise ValueError(
                    f"{path}: row {row_index}: unsupported schema_version {schema_version!r}"
                )
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
                row.get("protocol_json"), path=path, row_number=row_index
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
                for field in INPUT_LINEAGE_FIELDS
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
            for protocol_field, lineage_field in PROTOCOL_TO_LINEAGE.items():
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
    )
    if publication is not None:
        _validate_commit_marker(parsed, publication)
    return parsed


def _validate_repeat_provenance(parsed: list[ParsedEdgeFile], *, state: str) -> None:
    """Validate compatibility and independence metadata for one state."""
    if len({result.path.resolve() for result in parsed}) != len(parsed):
        raise ValueError(f"the same {state} result path was supplied more than once")
    if len({result.sha256 for result in parsed}) != len(parsed):
        raise ValueError(f"duplicate {state} result contents cannot be independent repeats")

    engines = {result.engine for result in parsed}
    if len(engines) != 1:
        raise ValueError(f"{state} repeats contain multiple engines: {sorted(engines)}")
    protocols = {result.protocol_fingerprint for result in parsed}
    if len(protocols) != 1:
        raise ValueError(f"{state} repeats contain multiple protocol fingerprints")
    for result in parsed[1:]:
        if result.input_lineage != parsed[0].input_lineage:
            baseline = parsed[0].input_lineage or {}
            current = result.input_lineage or {}
            differing = sorted(
                field for field in INPUT_LINEAGE_FIELDS if baseline.get(field) != current.get(field)
            )
            raise ValueError(f"{result.path}: repeat input lineage differs: {differing}")
    if len(parsed) > 1:
        metadata: dict[str, list[str | int | None]] = {
            "run seed": [result.run_seed for result in parsed],
            "run manifest": [result.run_manifest_sha256 for result in parsed],
            "run ID": [result.run_id for result in parsed],
            "source FMP": [result.source_fmp_sha256 for result in parsed],
            "vendor edge table": [result.vendor_edge_table_sha256 for result in parsed],
        }
        if (
            parsed[0].protocol_fingerprint is None
            or parsed[0].input_lineage is None
            or any(value is None for values in metadata.values() for value in values)
        ):
            raise ValueError(
                "repeat aggregation requires normalized exports with protocol, manifest, "
                "run-ID, seed, and source-FMP metadata"
            )
        for label, values in metadata.items():
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {state} {label}s are not independent repeats")


def _validate_repeat_inputs(parsed: list[ParsedEdgeFile], *, state: str) -> None:
    """Validate provenance and graph compatibility for one state's repeats."""
    _validate_repeat_provenance(parsed, state=state)

    expected_pairs = {(edge.source, edge.target) for edge in parsed[0].edges}
    expected_ligands = {ligand for edge in parsed[0].edges for ligand in (edge.source, edge.target)}
    for result in parsed[1:]:
        pairs = {(edge.source, edge.target) for edge in result.edges}
        ligands = {ligand for edge in result.edges for ligand in (edge.source, edge.target)}
        if pairs != expected_pairs:
            missing = sorted(expected_pairs - pairs)
            extra = sorted(pairs - expected_pairs)
            raise ValueError(
                f"{result.path}: repeat edge set differs; missing={missing}, extra={extra}"
            )
        if ligands != expected_ligands:
            raise ValueError(f"{result.path}: repeat ligand set differs")


def combine_repeats(
    paths: list[Path],
    *,
    state: str,
) -> tuple[list[CombinedEdge], list[ParsedEdgeFile]]:
    """Combine matching independent repeat edges by inverse-variance weighting."""
    if not paths:
        raise ValueError(f"at least one {state} result CSV is required")
    parsed = [read_edge_file(path, expected_state=state) for path in paths]
    _validate_repeat_inputs(parsed, state=state)
    expected_pairs = {(edge.source, edge.target) for edge in parsed[0].edges}

    by_pair: dict[tuple[str, str], list[Edge]] = {pair: [] for pair in expected_pairs}
    for result in parsed:
        for edge in result.edges:
            by_pair[(edge.source, edge.target)].append(edge)

    combined: list[CombinedEdge] = []
    for source, target in sorted(by_pair):
        observations = by_pair[(source, target)]
        weights = np.array([1.0 / edge.uncertainty**2 for edge in observations])
        values = np.array([edge.ddg for edge in observations])
        mean = float(np.dot(weights, values) / weights.sum())
        heterogeneity = float(np.dot(weights, (values - mean) ** 2))
        heterogeneity_dof = len(observations) - 1
        heterogeneity_p_value = (
            float(chi2.sf(heterogeneity, heterogeneity_dof)) if heterogeneity_dof > 0 else None
        )
        fixed_effect_uncertainty = float(math.sqrt(1.0 / weights.sum()))
        uncertainty_scale_factor = (
            math.sqrt(max(1.0, heterogeneity / heterogeneity_dof)) if heterogeneity_dof > 0 else 1.0
        )
        combined.append(
            CombinedEdge(
                source=source,
                target=target,
                ddg=mean,
                uncertainty=fixed_effect_uncertainty * uncertainty_scale_factor,
                repeat_count=len(observations),
                heterogeneity_chi2=heterogeneity,
                heterogeneity_dof=heterogeneity_dof,
                fixed_effect_uncertainty=fixed_effect_uncertainty,
                heterogeneity_p_value=heterogeneity_p_value,
                uncertainty_scale_factor=uncertainty_scale_factor,
            )
        )
    return combined, parsed


def _cycle_diagnostics(
    edges: list[CombinedEdge],
    *,
    ligands: list[str],
    reference: str,
) -> list[dict[str, Any]]:
    """Compute a deterministic fundamental-cycle basis from raw combined edges."""
    graph = nx.Graph()
    graph.add_nodes_from(ligands)
    graph.add_edges_from((edge.source, edge.target) for edge in edges)
    lookup = {(edge.source, edge.target): edge for edge in edges}
    cycles = nx.cycle_basis(graph, root=reference)
    normalized_cycles = sorted(
        cycles,
        key=lambda cycle: (len(cycle), tuple(sorted(cycle))),
    )
    diagnostics: list[dict[str, Any]] = []
    for cycle_index, cycle in enumerate(normalized_cycles, start=1):
        terms: list[dict[str, Any]] = []
        closure = 0.0
        variance = 0.0
        traversal = list(zip(cycle, cycle[1:] + cycle[:1]))
        for start, end in traversal:
            pair = (min(start, end), max(start, end))
            edge = lookup[pair]
            sign = 1.0 if (start, end) == pair else -1.0
            closure += sign * edge.ddg
            variance += edge.uncertainty**2
            terms.append({"source": start, "target": end, "sign": int(sign)})
        standard_uncertainty = math.sqrt(variance)
        diagnostics.append({
            "cycle_id": f"cycle_{cycle_index:03d}",
            "nodes": cycle,
            "terms": terms,
            "closure_kcal_mol": closure,
            "standard_uncertainty_kcal_mol": standard_uncertainty,
            "z_score": closure / standard_uncertainty,
        })
    return diagnostics


def fit_network(
    edges: list[CombinedEdge],
    *,
    reference: str,
) -> dict[str, Any]:
    """Fit one connected RBFE network using diagonal-error weighted least squares."""
    ligands = sorted({ligand for edge in edges for ligand in (edge.source, edge.target)})
    if reference not in ligands:
        raise ValueError(f"reference ligand {reference!r} is absent from the network")
    ligand_index = {ligand: index for index, ligand in enumerate(ligands)}
    incidence = np.zeros((len(edges), len(ligands)), dtype=np.float64)
    observed = np.empty(len(edges), dtype=np.float64)
    uncertainty = np.empty(len(edges), dtype=np.float64)
    for row, edge in enumerate(edges):
        incidence[row, ligand_index[edge.source]] = -1.0
        incidence[row, ligand_index[edge.target]] = 1.0
        observed[row] = edge.ddg
        uncertainty[row] = edge.uncertainty

    reference_index = ligand_index[reference]
    reduced = np.delete(incidence, reference_index, axis=1)
    whitened_design = reduced / uncertainty[:, None]
    whitened_observed = observed / uncertainty
    estimate_reduced, _, rank, singular_values = np.linalg.lstsq(
        whitened_design,
        whitened_observed,
        rcond=None,
    )
    if rank != len(ligands) - 1:
        raise ValueError("network design matrix is rank deficient")
    _, singular_values_svd, vh = np.linalg.svd(whitened_design, full_matrices=False)
    covariance_reduced = (vh.T * (1.0 / singular_values_svd**2)) @ vh

    estimate = np.insert(estimate_reduced, reference_index, 0.0)
    covariance = np.zeros((len(ligands), len(ligands)), dtype=np.float64)
    keep = [index for index in range(len(ligands)) if index != reference_index]
    covariance[np.ix_(keep, keep)] = covariance_reduced
    fitted = incidence @ estimate
    residual = observed - fitted
    standardized_residual = residual / uncertainty
    chi_square = float(np.dot(standardized_residual, standardized_residual))
    degrees_of_freedom = len(edges) - len(ligands) + 1
    p_value = float(chi2.sf(chi_square, degrees_of_freedom)) if degrees_of_freedom > 0 else None
    repeat_chi_square = sum(edge.heterogeneity_chi2 for edge in edges)
    repeat_degrees_of_freedom = sum(edge.heterogeneity_dof for edge in edges)
    repeat_p_value = (
        float(chi2.sf(repeat_chi_square, repeat_degrees_of_freedom))
        if repeat_degrees_of_freedom > 0
        else None
    )
    residuals = []
    for edge, predicted, difference, z_score in zip(edges, fitted, residual, standardized_residual):
        residuals.append({
            "source": edge.source,
            "target": edge.target,
            "observed_kcal_mol": edge.ddg,
            "fitted_kcal_mol": float(predicted),
            "residual_kcal_mol": float(difference),
            "standard_uncertainty_kcal_mol": edge.uncertainty,
            "fixed_effect_standard_uncertainty_kcal_mol": (
                edge.fixed_effect_uncertainty
                if edge.fixed_effect_uncertainty is not None
                else edge.uncertainty
            ),
            "residual_over_input_uncertainty": float(z_score),
            "repeat_count": edge.repeat_count,
            "repeat_heterogeneity_chi2": edge.heterogeneity_chi2,
            "repeat_heterogeneity_dof": edge.heterogeneity_dof,
            "repeat_heterogeneity_p_value": edge.heterogeneity_p_value,
            "repeat_uncertainty_scale_factor": edge.uncertainty_scale_factor,
        })

    return {
        "ligands": ligands,
        "reference": reference,
        "relative_free_energy_kcal_mol": estimate,
        "covariance_kcal2_mol2": covariance,
        "standard_uncertainty_kcal_mol": np.sqrt(np.diag(covariance)),
        "diagnostics": {
            "chi_square": chi_square,
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": p_value,
            "p_value_is_approximate": True,
            "cycle_consistency_assessable": degrees_of_freedom > 0,
            "repeat_heterogeneity_chi_square": repeat_chi_square,
            "repeat_heterogeneity_degrees_of_freedom": repeat_degrees_of_freedom,
            "repeat_heterogeneity_p_value": repeat_p_value,
            "repeat_uncertainty_policy": "edgewise_birge_ratio_at_least_one",
            "design_rank": rank,
            "design_singular_values": singular_values.tolist(),
        },
        "edge_residuals": residuals,
        "cycles": _cycle_diagnostics(edges, ligands=ligands, reference=reference),
    }


def _read_anchor(path: Path, *, reference: str) -> dict[str, Any]:
    """Read and validate an independent absolute anchor."""
    anchor = json.loads(path.read_text())
    if not isinstance(anchor, dict):
        raise ValueError(f"{path}: anchor must be a JSON object")
    required = {
        "reference",
        "quantity",
        "sign_convention",
        "estimate",
        "standard_uncertainty",
        "unit",
    }
    missing = sorted(required - anchor.keys())
    if missing:
        raise ValueError(f"{path}: anchor is missing fields: {missing}")
    if anchor["reference"] != reference:
        raise ValueError(
            f"{path}: anchor reference {anchor['reference']!r} does not match {reference!r}"
        )
    if anchor["quantity"] not in ANCHOR_QUANTITIES:
        raise ValueError(f"{path}: unsupported anchor quantity {anchor['quantity']!r}")
    if anchor["sign_convention"] != "open_minus_closed":
        raise ValueError(f"{path}: anchor sign must be 'open_minus_closed'")
    unit = anchor["unit"]
    if unit not in SUPPORTED_UNITS:
        raise ValueError(f"{path}: unsupported anchor unit {unit!r}")
    try:
        estimate = float(anchor["estimate"])
        uncertainty = float(anchor["standard_uncertainty"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: anchor estimate and uncertainty must be numeric") from exc
    if not math.isfinite(estimate) or not math.isfinite(uncertainty):
        raise ValueError(f"{path}: anchor estimate and uncertainty must be finite")
    if uncertainty <= 0.0:
        raise ValueError(f"{path}: anchor uncertainty must be greater than zero")
    factor = SUPPORTED_UNITS[unit]
    return {
        "reference": reference,
        "quantity": anchor["quantity"],
        "sign_convention": "open_minus_closed",
        "estimate_kcal_mol": estimate * factor,
        "standard_uncertainty_kcal_mol": uncertainty * factor,
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _converted_fit(fit: dict[str, Any], *, factor: float) -> dict[str, Any]:
    """Convert a fit dictionary from internal kcal/mol to the output unit."""
    return {
        "relative_free_energy": (fit["relative_free_energy_kcal_mol"] * factor).tolist(),
        "standard_uncertainty": (fit["standard_uncertainty_kcal_mol"] * factor).tolist(),
        "covariance": (fit["covariance_kcal2_mol2"] * factor**2).tolist(),
        "diagnostics": fit["diagnostics"],
    }


def _input_lineage(result: ParsedEdgeFile) -> dict[str, str]:
    """Return required normalized input lineage after schema validation."""
    if result.input_lineage is None:
        raise ValueError(f"{result.path}: normalized input lineage is missing")
    return result.input_lineage


def _quality_control(result: ParsedEdgeFile) -> dict[str, Any]:
    """Return required normalized vendor QC after schema validation."""
    if result.quality_control is None:
        raise ValueError(f"{result.path}: normalized vendor QC is missing")
    return result.quality_control


def _validate_paired_inputs(
    open_files: list[ParsedEdgeFile],
    closed_files: list[ParsedEdgeFile],
) -> list[ParsedEdgeFile]:
    """Validate provenance compatibility and independence across states."""
    all_files = [*open_files, *closed_files]
    if any(result.schema != "normalized_v2" for result in all_files) or any(
        value is None
        for result in all_files
        for value in (
            result.protocol_fingerprint,
            result.protocol_json,
            result.run_manifest_sha256,
            result.run_id,
            result.run_seed,
            result.source_fmp_sha256,
            result.input_lineage,
            result.vendor_edge_table_sha256,
            result.quality_control,
        )
    ):
        raise ValueError(
            "paired-state analysis requires normalized exports from extract_fep_results.py "
            "with run-manifest provenance"
        )
    if {result.sha256 for result in open_files} & {result.sha256 for result in closed_files}:
        raise ValueError("identical result content was supplied to both open and closed states")

    cross_state_identities: dict[str, tuple[set[str | None], set[str | None]]] = {
        "source FMP": (
            {result.source_fmp_sha256 for result in open_files},
            {result.source_fmp_sha256 for result in closed_files},
        ),
        "run manifest": (
            {result.run_manifest_sha256 for result in open_files},
            {result.run_manifest_sha256 for result in closed_files},
        ),
        "run ID": (
            {result.run_id for result in open_files},
            {result.run_id for result in closed_files},
        ),
        "input map": (
            {_input_lineage(result)["input_map_sha256"] for result in open_files},
            {_input_lineage(result)["input_map_sha256"] for result in closed_files},
        ),
    }
    for label, (open_values, closed_values) in cross_state_identities.items():
        if open_values & closed_values:
            raise ValueError(f"the same {label} was assigned to both open and closed states")
    shared_lineage_fields = (
        "input_edge_sha256",
        "input_atom_mapping_fingerprint",
        "input_ligand_bundle_sha256",
        "input_ligand_validation_sha256",
        "input_receptor_microstates_sha256",
        "input_paired_inputs_sha256",
    )
    for field in shared_lineage_fields:
        values = {_input_lineage(result)[field] for result in all_files}
        if len(values) != 1:
            raise ValueError(f"open/closed runs have incompatible {field}")
    open_engines = {result.engine for result in open_files}
    closed_engines = {result.engine for result in closed_files}
    if open_engines != closed_engines:
        raise ValueError(
            f"open/closed result engines differ: open={sorted(open_engines)}, "
            f"closed={sorted(closed_engines)}"
        )
    if len({result.protocol_fingerprint for result in all_files}) != 1:
        raise ValueError("open/closed runs have different scientific protocol fingerprints")
    all_seeds = [result.run_seed for result in all_files]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("run seeds must be unique across all open and closed repeats")
    return all_files


def analyze(
    open_paths: list[Path],
    closed_paths: list[Path],
    *,
    reference: str,
    output_unit: str = "kcal/mol",
    anchor_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Analyze paired open/closed result networks.

    Args:
        open_paths: One normalized schema-v2 CSV per independent open repeat.
        closed_paths: One normalized schema-v2 CSV per independent closed repeat.
        reference: Ligand defining the relative-free-energy gauge.
        output_unit: ``kcal/mol`` or ``kJ/mol``.
        anchor_path: Optional independent reference anchor JSON.

    Returns:
        ``(analysis, tables)`` where ``analysis`` is JSON-serializable and
        ``tables`` contains detailed edge and cycle diagnostic rows.

    Raises:
        ValueError: If inputs are invalid or the two networks are incomparable.
    """
    if output_unit not in SUPPORTED_UNITS:
        raise ValueError(f"unsupported output unit {output_unit!r}")
    open_edges, open_files = combine_repeats(open_paths, state="open")
    closed_edges, closed_files = combine_repeats(closed_paths, state="closed")
    all_files = _validate_paired_inputs(open_files, closed_files)
    open_fit = fit_network(open_edges, reference=reference)
    closed_fit = fit_network(closed_edges, reference=reference)
    if open_fit["ligands"] != closed_fit["ligands"]:
        open_only = sorted(set(open_fit["ligands"]) - set(closed_fit["ligands"]))
        closed_only = sorted(set(closed_fit["ligands"]) - set(open_fit["ligands"]))
        raise ValueError(
            f"open/closed ligand sets differ: open_only={open_only}, closed_only={closed_only}"
        )

    ligands: list[str] = open_fit["ligands"]
    relative = (
        open_fit["relative_free_energy_kcal_mol"] - closed_fit["relative_free_energy_kcal_mol"]
    )
    relative_covariance = open_fit["covariance_kcal2_mol2"] + closed_fit["covariance_kcal2_mol2"]
    factor = 1.0 / SUPPORTED_UNITS[output_unit]
    anchored: dict[str, Any] | None = None
    if anchor_path is not None:
        anchor = _read_anchor(anchor_path, reference=reference)
        anchor_value = anchor["estimate_kcal_mol"]
        anchor_variance = anchor["standard_uncertainty_kcal_mol"] ** 2
        anchored_values = relative + anchor_value
        anchored_covariance = relative_covariance + anchor_variance * np.ones(
            relative_covariance.shape
        )
        anchored = {
            "quantity": anchor["quantity"],
            "sign_convention": "open_minus_closed",
            "value": (anchored_values * factor).tolist(),
            "standard_uncertainty": (np.sqrt(np.diag(anchored_covariance)) * factor).tolist(),
            "covariance": (anchored_covariance * factor**2).tolist(),
            "anchor": {
                "reference": reference,
                "estimate": anchor_value * factor,
                "standard_uncertainty": math.sqrt(anchor_variance) * factor,
                "source": anchor["source"],
                "sha256": anchor["sha256"],
            },
        }

    def provenance(files: list[ParsedEdgeFile]) -> list[dict[str, Any]]:
        return [
            {
                "path": str(result.path),
                "sha256": result.sha256,
                "schema": result.schema,
                "engine": result.engine,
                "protocol_fingerprint": result.protocol_fingerprint,
                "run_manifest_sha256": result.run_manifest_sha256,
                "run_id": result.run_id,
                "run_seed": result.run_seed,
                "source_fmp_sha256": result.source_fmp_sha256,
                "input_lineage": result.input_lineage,
                "vendor_edge_table_sha256": result.vendor_edge_table_sha256,
                "quality_control": result.quality_control,
            }
            for result in files
        ]

    protocol_json = all_files[0].protocol_json
    assert protocol_json is not None
    scientific_protocol = json.loads(protocol_json)
    quality_control_runs = {
        "open": [
            {"run_id": result.run_id, "summary": _quality_control(result)} for result in open_files
        ],
        "closed": [
            {"run_id": result.run_id, "summary": _quality_control(result)}
            for result in closed_files
        ],
    }
    qc_warning_count = sum(_quality_control(result)["warning_edge_count"] for result in all_files)
    assumptions = [
        "raw directed edge ddG equals G(target) - G(source)",
        "reported edge uncertainties are one-standard-uncertainty sampling errors",
        "edge errors are independent within each network (diagonal covariance)",
        "open and closed simulations are independent",
        "shared force-field and other systematic errors are not represented",
        "independent repeats use a fixed-effect inverse-variance mean",
        "repeat disagreement inflates each pooled edge uncertainty by the Birge ratio",
        "chi-square p-values are approximate and assume normal calibrated input errors",
        "vendor QC ratings are reported diagnostically and do not exclude edges",
    ]
    if anchored is not None:
        assumptions.append("the external anchor is independent of both RBFE networks")
    if scientific_protocol["input_allow_microstate_mismatch"] == "1":
        assumptions.append("the receptor microstate mismatch allowance was enabled")
    analysis = {
        "schema_version": 2,
        "quantity": "reference_relative_binding_selectivity_open_minus_closed",
        "sign_convention": (
            "open_minus_closed; negative means more open-selective than the reference"
        ),
        "reference": reference,
        "unit": output_unit,
        "ligands": ligands,
        "scientific_protocol": scientific_protocol,
        "protocol_fingerprint": all_files[0].protocol_fingerprint,
        "quality_control": {
            "has_warnings": qc_warning_count > 0,
            "warning_edge_ratings_across_runs": qc_warning_count,
            "runs": quality_control_runs,
            "policy": "diagnostic_only_no_automatic_edge_exclusion",
        },
        "assumptions": assumptions,
        "input_provenance": {
            "open": provenance(open_files),
            "closed": provenance(closed_files),
        },
        "open": _converted_fit(open_fit, factor=factor),
        "closed": _converted_fit(closed_fit, factor=factor),
        "relative_selectivity": {
            "value": (relative * factor).tolist(),
            "standard_uncertainty": (np.sqrt(np.diag(relative_covariance)) * factor).tolist(),
            "covariance": (relative_covariance * factor**2).tolist(),
        },
        "anchored": anchored,
    }
    tables = {
        "open_edge_residuals": open_fit["edge_residuals"],
        "closed_edge_residuals": closed_fit["edge_residuals"],
        "open_cycles": open_fit["cycles"],
        "closed_cycles": closed_fit["cycles"],
    }
    return analysis, tables


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of flat diagnostic dictionaries to CSV."""
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar values to their JSON-native equivalents."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_output_files(
    analysis: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
) -> None:
    """Write one unpublished analysis bundle to an empty staging directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False, default=_json_default) + "\n"
    )
    unit = analysis["unit"]
    selectivity_rows: list[dict[str, Any]] = []
    anchored = analysis["anchored"]
    for index, ligand in enumerate(analysis["ligands"]):
        row = {
            "ligand": ligand,
            "reference": analysis["reference"],
            "unit": unit,
            "open_relative_dg": analysis["open"]["relative_free_energy"][index],
            "open_standard_uncertainty": analysis["open"]["standard_uncertainty"][index],
            "closed_relative_dg": analysis["closed"]["relative_free_energy"][index],
            "closed_standard_uncertainty": analysis["closed"]["standard_uncertainty"][index],
            "relative_open_minus_closed": analysis["relative_selectivity"]["value"][index],
            "relative_standard_uncertainty": analysis["relative_selectivity"][
                "standard_uncertainty"
            ][index],
        }
        if anchored is not None:
            row["anchored_quantity"] = anchored["quantity"]
            row["anchored_open_minus_closed"] = anchored["value"][index]
            row["anchored_standard_uncertainty"] = anchored["standard_uncertainty"][index]
        selectivity_rows.append(row)
    _write_rows(output_dir / "selectivity.csv", selectivity_rows)

    factor = 1.0 / SUPPORTED_UNITS[unit]
    for name, rows in tables.items():
        converted_rows: list[dict[str, Any]] = []
        for original in rows:
            row = dict(original)
            for key in list(row):
                if key.endswith("_kcal_mol"):
                    row[key.removesuffix("_kcal_mol")] = row.pop(key) * factor
            if "terms" in row:
                row["terms"] = json.dumps(row["terms"], separators=(",", ":"))
            if "nodes" in row:
                row["nodes"] = " -> ".join([*row["nodes"], row["nodes"][0]])
            row["unit"] = unit
            converted_rows.append(row)
        _write_rows(output_dir / f"{name}.csv", converted_rows)


def write_outputs(
    analysis: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
) -> None:
    """Stage and publish a locked analysis bundle with a commit marker."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f"{output_dir.name}.lock")
    marker_name = "analysis.provenance.json"
    expected_names = {
        "analysis.json",
        "selectivity.csv",
        "open_edge_residuals.csv",
        "closed_edge_residuals.csv",
        "open_cycles.csv",
        "closed_cycles.csv",
    }
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ) as staging_name:
        staging_dir = Path(staging_name)
        _write_output_files(analysis, tables, output_dir=staging_dir)
        staged_names = {path.name for path in staging_dir.iterdir() if path.is_file()}
        if staged_names != expected_names:
            missing = sorted(expected_names - staged_names)
            extra = sorted(staged_names - expected_names)
            raise ValueError(
                f"analysis output bundle is incomplete; missing={missing}, extra={extra}"
            )
        marker = {
            "schema_version": 1,
            "publication_status": "complete_commit_marker",
            "reference": analysis["reference"],
            "protocol_fingerprint": analysis["protocol_fingerprint"],
            "files": {
                name: hashlib.sha256((staging_dir / name).read_bytes()).hexdigest()
                for name in sorted(expected_names)
            },
        }
        staged_marker = staging_dir / marker_name
        staged_marker.write_text(json.dumps(marker, indent=2) + "\n")

        with lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            output_dir.mkdir(parents=True, exist_ok=True)
            published_marker = output_dir / marker_name
            published_marker.unlink(missing_ok=True)
            for name in sorted(expected_names):
                os.replace(staging_dir / name, output_dir / name)
            os.replace(staged_marker, published_marker)


def main() -> None:
    """Run the command-line analyzer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open",
        type=Path,
        action="append",
        required=True,
        help="Open-state edge CSV; repeat the option for independent repeats.",
    )
    parser.add_argument(
        "--closed",
        type=Path,
        action="append",
        required=True,
        help="Closed-state edge CSV; repeat the option for independent repeats.",
    )
    parser.add_argument("--reference", required=True, help="Explicit reference ligand title.")
    parser.add_argument(
        "--unit",
        choices=tuple(SUPPORTED_UNITS),
        default="kcal/mol",
        help="Output energy unit (default: kcal/mol).",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        help="Optional independent absolute open-minus-closed reference anchor JSON.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for analysis.json and diagnostic CSVs.",
    )
    args = parser.parse_args()
    try:
        analysis, tables = analyze(
            args.open,
            args.closed,
            reference=args.reference,
            output_unit=args.unit,
            anchor_path=args.anchor,
        )
        write_outputs(analysis, tables, output_dir=args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"wrote FEP+ open/closed analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
