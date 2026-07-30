#!/usr/bin/env python3
"""Export raw Bennett edge results from one completed FEP+ ``*_out.fmp``.

This is a strict adapter around Schrodinger's supported ``fmp2excel.py``.  It
rejects blank pre-run maps and writes a small, versioned CSV consumed by
``analyze_fep_results.py``.  Cycle-closure-adjusted values are retained only in
the vendor's temporary export and are not copied into the statistical input.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from analyze_fep_results import ParsedEdgeFile, read_edge_file

PROTOCOL_FIELDS = (
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
)
SNAPSHOT_HASH_FIELDS = {
    "map_sha256": "input_map.fmp",
    "edge_sha256": "input_map.edge",
    "map_provenance_sha256": "input_map.provenance",
    "atom_mapping_json_sha256": "input_map_mappings.json",
    "ligand_bundle_sha256": "input_ligands.maegz",
    "ligand_validation_sha256": "input_ligand_validation.json",
    "receptor_microstates_sha256": "input_receptor_microstates.json",
    "paired_inputs_sha256": "input_paired_inputs.tsv",
}
QC_VENDOR_FIELDS = {
    "qc_convergence": "convergence",
    "qc_ligand_rmsd": "ligand RMSD",
    "qc_rest_exchange": "REST exchange",
    "qc_ccc_convergence": "CCC Conv",
}
QC_RATINGS = {"Good", "Fair", "Bad", "N/A"}
EDGE_LINE = re.compile(r"^\s*\S+:\S+\s*#\s*(?P<a>\S+)\s*->\s*(?P<b>\S+)\s*$")
MAX_SEED = 2147483647
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "state",
    "seed",
    "jobname",
    "prepare_only",
    *PROTOCOL_FIELDS,
    *SNAPSHOT_HASH_FIELDS,
}
NORMALIZED_FIELDS = [
    "schema_version",
    "engine",
    "protocol_fingerprint",
    "protocol_json",
    "run_manifest_sha256",
    "run_id",
    "run_seed",
    "source_fmp_sha256",
    "input_map_sha256",
    "input_edge_sha256",
    "input_map_provenance_sha256",
    "input_atom_mapping_fingerprint",
    "input_atom_mapping_json_sha256",
    "input_ligand_bundle_sha256",
    "input_ligand_validation_sha256",
    "input_receptor_microstates_sha256",
    "input_paired_inputs_sha256",
    "vendor_edge_table_sha256",
    *QC_VENDOR_FIELDS,
    "state",
    "edge_id",
    "source",
    "target",
    "ddg",
    "standard_uncertainty",
    "unit",
    "sign_convention",
]


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Validated run-manifest metadata used by normalized results."""

    manifest_sha256: str
    run_id: str
    seed: int
    protocol: dict[str, str]
    protocol_json: str
    protocol_fingerprint: str
    suite_release: str
    input_lineage: dict[str, str]
    expected_edge_pairs: frozenset[tuple[str, str]]
    expected_mapping_fingerprint: str


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _suite_release(schrodinger: Path) -> str:
    """Return a whitespace-normalized Suite release string."""
    version_file = schrodinger / "version.txt"
    if not version_file.is_file():
        raise FileNotFoundError(f"Suite release metadata not found: {version_file}")
    release = " ".join(version_file.read_text().split())
    if not release:
        raise ValueError(f"Suite release metadata is empty: {version_file}")
    return release


def _is_sha256(value: str) -> bool:
    """Return whether a string is a lowercase SHA-256 digest."""
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _read_manifest_values(manifest: Path) -> tuple[dict[str, str], bytes]:
    """Parse the launcher's strict two-column TSV representation."""
    if not manifest.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest}")
    raw_bytes = manifest.read_bytes()
    try:
        text = raw_bytes.decode()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{manifest}: manifest is not valid UTF-8") from exc
    values: dict[str, str] = {}
    for row_number, line in enumerate(text.splitlines(), start=1):
        columns = line.split("\t")
        if len(columns) != 2 or not all(column.strip() for column in columns):
            raise ValueError(f"{manifest}: row {row_number}: expected two nonempty TSV fields")
        key, value = (column.strip() for column in columns)
        if key in values:
            raise ValueError(f"{manifest}: row {row_number}: duplicate key {key!r}")
        values[key] = value
    missing = sorted(REQUIRED_MANIFEST_FIELDS - values.keys())
    if missing:
        raise ValueError(f"{manifest}: missing required fields: {missing}")
    return values, raw_bytes


def _read_paired_input_values(path: Path) -> dict[str, str]:
    """Read the builder's strict shared open/closed cohort manifest."""
    required = {
        "schema_version",
        "build_schema",
        "suite_release",
        "allow_microstate_mismatch",
        "open_receptor_sha256",
        "closed_receptor_sha256",
        "ligand_bundle_sha256",
        "ligand_validation_sha256",
        "receptor_microstates_sha256",
        "open_map_sha256",
        "closed_map_sha256",
        "open_edge_sha256",
        "closed_edge_sha256",
        "open_map_provenance_sha256",
        "closed_map_provenance_sha256",
        "open_atom_mapping_fingerprint",
        "closed_atom_mapping_fingerprint",
    }
    values: dict[str, str] = {}
    for row_number, line in enumerate(path.read_text().splitlines(), start=1):
        columns = line.split("\t")
        if len(columns) != 2 or not all(column.strip() for column in columns):
            raise ValueError(f"{path}: row {row_number}: expected two nonempty TSV fields")
        key, value = (column.strip() for column in columns)
        if key in values:
            raise ValueError(f"{path}: row {row_number}: duplicate key {key!r}")
        values[key] = value
    missing = sorted(required - values.keys())
    extra = sorted(values.keys() - required)
    if missing or extra:
        raise ValueError(f"{path}: paired-input fields differ; missing={missing}, extra={extra}")
    if values["schema_version"] != "1":
        raise ValueError(f"{path}: unsupported schema_version {values['schema_version']!r}")
    hash_fields = required - {
        "schema_version",
        "build_schema",
        "suite_release",
        "allow_microstate_mismatch",
    }
    for field in hash_fields:
        if not _is_sha256(values[field]):
            raise ValueError(f"{path}: {field} must be a lowercase SHA-256")
    return values


def _canonical_pair(source: str, target: str, *, context: str) -> tuple[str, str]:
    """Return one validated unordered ligand pair."""
    if not source or not target:
        raise ValueError(f"{context}: ligand names must be nonempty")
    if source == target:
        raise ValueError(f"{context}: self-edge is not valid: {source!r}")
    return (source, target) if source < target else (target, source)


def _read_mapping_pairs(
    mapping_path: Path,
    *,
    expected_fingerprint: str,
) -> set[tuple[str, str]]:
    """Read the unique ligand pairs from one atom-mapping snapshot."""
    try:
        mapping = json.loads(mapping_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{mapping_path}: invalid mapping JSON") from exc
    if not isinstance(mapping, dict) or mapping.get("schema_version") != 2:
        raise ValueError(f"{mapping_path}: expected mapping schema_version 2")
    mapping_fingerprint = mapping.get("mapping_fingerprint")
    if mapping_fingerprint != expected_fingerprint:
        raise ValueError(f"{mapping_path}: atom mapping fingerprint does not match its manifest")
    records = mapping.get("edges")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{mapping_path}: mapping edge list must be nonempty")
    mapping_pairs: set[tuple[str, str]] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{mapping_path}: mapping edge {index} must be an object")
        source = record.get("name_a")
        target = record.get("name_b")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"{mapping_path}: mapping edge {index} has invalid ligand names")
        pair = _canonical_pair(
            source.strip(), target.strip(), context=f"{mapping_path}: mapping edge {index}"
        )
        if pair in mapping_pairs:
            raise ValueError(f"{mapping_path}: duplicate mapping ligand pair {pair}")
        mapping_pairs.add(pair)
    return mapping_pairs


def _read_edge_pairs(edge_path: Path) -> set[tuple[str, str]]:
    """Read the unique ligand pairs from one strict FEP+ edge snapshot."""
    edge_pairs: set[tuple[str, str]] = set()
    for line_number, line in enumerate(edge_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        match = EDGE_LINE.match(line)
        if match is None:
            raise ValueError(f"{edge_path}: malformed edge line {line_number}: {line!r}")
        pair = _canonical_pair(
            match["a"], match["b"], context=f"{edge_path}: edge line {line_number}"
        )
        if pair in edge_pairs:
            raise ValueError(f"{edge_path}: duplicate ligand pair {pair}")
        edge_pairs.add(pair)
    if not edge_pairs:
        raise ValueError(f"{edge_path}: edge snapshot is empty")
    return edge_pairs


def _read_snapshot_edge_pairs(manifest: Path, values: dict[str, str]) -> frozenset[tuple[str, str]]:
    """Cross-check exact edge and mapping snapshots and return their ligand pairs."""
    mapping_path = manifest.parent / SNAPSHOT_HASH_FIELDS["atom_mapping_json_sha256"]
    mapping_pairs = _read_mapping_pairs(
        mapping_path,
        expected_fingerprint=values["atom_mapping_fingerprint"],
    )
    edge_path = manifest.parent / SNAPSHOT_HASH_FIELDS["edge_sha256"]
    edge_pairs = _read_edge_pairs(edge_path)
    if edge_pairs != mapping_pairs:
        missing = sorted(mapping_pairs - edge_pairs)
        extra = sorted(edge_pairs - mapping_pairs)
        raise ValueError(
            f"{manifest}: edge and mapping snapshots differ; missing={missing}, extra={extra}"
        )
    return frozenset(edge_pairs)


def _validate_snapshot_lineage(
    manifest: Path,
    values: dict[str, str],
    *,
    state: str,
    suite_release: str,
) -> tuple[dict[str, str], frozenset[tuple[str, str]]]:
    """Verify exact run snapshots and return lineage fields plus edge pairs."""
    for field in {*SNAPSHOT_HASH_FIELDS, "atom_mapping_fingerprint"}:
        if not _is_sha256(values[field]):
            raise ValueError(f"{manifest}: {field} must be a lowercase SHA-256")
    for field, filename in SNAPSHOT_HASH_FIELDS.items():
        snapshot = manifest.parent / filename
        if not snapshot.is_file():
            raise FileNotFoundError(f"run snapshot not found: {snapshot}")
        if _sha256(snapshot) != values[field]:
            raise ValueError(f"{manifest}: {field} does not match {snapshot.name}")
    paired_path = manifest.parent / SNAPSHOT_HASH_FIELDS["paired_inputs_sha256"]
    paired = _read_paired_input_values(paired_path)
    if " ".join(paired["suite_release"].split()) != suite_release:
        raise ValueError(f"{paired_path}: Suite release does not match the run manifest")
    expected_paired_values = {
        "allow_microstate_mismatch": values["input_allow_microstate_mismatch"],
        "ligand_bundle_sha256": values["ligand_bundle_sha256"],
        "ligand_validation_sha256": values["ligand_validation_sha256"],
        "receptor_microstates_sha256": values["receptor_microstates_sha256"],
        f"{state}_map_sha256": values["map_sha256"],
        f"{state}_edge_sha256": values["edge_sha256"],
        f"{state}_map_provenance_sha256": values["map_provenance_sha256"],
        f"{state}_atom_mapping_fingerprint": values["atom_mapping_fingerprint"],
    }
    for field, expected in expected_paired_values.items():
        if paired[field] != expected:
            raise ValueError(f"{paired_path}: {field} does not match the run snapshot")
    if paired["open_edge_sha256"] != paired["closed_edge_sha256"]:
        raise ValueError(f"{paired_path}: open/closed edge hashes differ")
    if paired["open_atom_mapping_fingerprint"] != paired["closed_atom_mapping_fingerprint"]:
        raise ValueError(f"{paired_path}: open/closed atom-mapping fingerprints differ")
    expected_edge_pairs = _read_snapshot_edge_pairs(manifest, values)
    lineage = {
        "input_map_sha256": values["map_sha256"],
        "input_edge_sha256": values["edge_sha256"],
        "input_map_provenance_sha256": values["map_provenance_sha256"],
        "input_atom_mapping_fingerprint": values["atom_mapping_fingerprint"],
        "input_atom_mapping_json_sha256": values["atom_mapping_json_sha256"],
        "input_ligand_bundle_sha256": values["ligand_bundle_sha256"],
        "input_ligand_validation_sha256": values["ligand_validation_sha256"],
        "input_receptor_microstates_sha256": values["receptor_microstates_sha256"],
        "input_paired_inputs_sha256": values["paired_inputs_sha256"],
    }
    return lineage, expected_edge_pairs


def _read_vendor_qc(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read official categorical per-edge QC ratings, mapping blanks to N/A."""
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        qc_by_pair: dict[tuple[str, str], dict[str, str]] = {}
        for row_number, row in enumerate(rows, start=2):
            source = (row.get("Ligand1") or "").strip()
            target = (row.get("Ligand2") or "").strip()
            if not source or not target:
                raise ValueError(f"{path}: row {row_number}: missing QC edge ligand name")
            pair = (min(source, target), max(source, target))
            ratings: dict[str, str] = {}
            for normalized_field, vendor_field in QC_VENDOR_FIELDS.items():
                rating = (row.get(vendor_field) or "").strip() or "N/A"
                if rating not in QC_RATINGS:
                    raise ValueError(
                        f"{path}: row {row_number}: unsupported {vendor_field} rating {rating!r}"
                    )
                ratings[normalized_field] = rating
            if pair in qc_by_pair:
                raise ValueError(f"{path}: duplicate QC ligand pair {pair}")
            qc_by_pair[pair] = ratings
    return qc_by_pair


def _read_manifest(
    manifest: Path,
    *,
    state: str,
    fmp: Path,
    schrodinger: Path,
) -> RunMetadata:
    """Read and validate scientific run identity from a launcher manifest."""
    values, raw_bytes = _read_manifest_values(manifest)
    if values["schema_version"] != "1":
        raise ValueError(f"{manifest}: unsupported schema_version {values['schema_version']!r}")
    if values["state"] != state:
        raise ValueError(
            f"{manifest}: state {values['state']!r} does not match requested {state!r}"
        )
    if values["prepare_only"] != "0":
        raise ValueError(f"{manifest}: prepare-only runs cannot be analyzed")
    try:
        seed = int(values["seed"])
    except ValueError as exc:
        raise ValueError(f"{manifest}: seed must be an integer") from exc
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"{manifest}: seed must be in [0, {MAX_SEED}]")
    run_id = values["jobname"]
    expected_fmp_name = f"{run_id}_out.fmp"
    if fmp.name != expected_fmp_name:
        raise ValueError(
            f"{manifest}: jobname {run_id!r} expects {expected_fmp_name!r}, got {fmp.name!r}"
        )
    suite_release = _suite_release(schrodinger)
    if " ".join(values["suite_release"].split()) != suite_release:
        raise ValueError(
            f"{manifest}: Suite release differs from extraction installation: "
            f"{values['suite_release']!r} != {suite_release!r}"
        )
    protocol = {field: values[field] for field in PROTOCOL_FIELDS}
    protocol["suite_release"] = suite_release
    input_lineage, expected_edge_pairs = _validate_snapshot_lineage(
        manifest,
        values,
        state=state,
        suite_release=suite_release,
    )
    protocol_json = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return RunMetadata(
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        run_id=run_id,
        seed=seed,
        protocol=protocol,
        protocol_json=protocol_json,
        protocol_fingerprint=hashlib.sha256(protocol_json.encode()).hexdigest(),
        suite_release=suite_release,
        input_lineage=input_lineage,
        expected_edge_pairs=expected_edge_pairs,
        expected_mapping_fingerprint=values["atom_mapping_fingerprint"],
    )


def _validate_extract_request(
    fmp: Path,
    output: Path,
    *,
    state: str,
    schrodinger: Path,
    manifest: Path,
) -> Path:
    """Validate extraction paths and return the Suite runner."""
    if state not in {"open", "closed"}:
        raise ValueError(f"state must be 'open' or 'closed', got {state!r}")
    if not fmp.is_file():
        raise FileNotFoundError(f"FEP+ output map not found: {fmp}")
    if output.suffix.lower() != ".csv":
        raise ValueError(f"normalized output must use a .csv suffix: {output}")
    if output.resolve() in {fmp.resolve(), manifest.resolve()}:
        raise ValueError("normalized output must differ from the FMP and run manifest")
    runner = schrodinger / "run"
    if not runner.is_file():
        raise FileNotFoundError(f"Schrodinger runner not found: {runner}")
    return runner


def _export_vendor_table(
    fmp: Path,
    *,
    runner: Path,
    export_root: Path,
    source_fmp_sha256: str,
) -> Path:
    """Run the official exporter and return its raw Bennett edge table."""
    command = [
        str(runner),
        "-FROM",
        "scisol",
        "fmp2excel.py",
        "-csv",
        "-cycle-closure",
        "-o",
        str(export_root),
        str(fmp),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"fmp2excel.py failed with exit code {completed.returncode}: {detail}")
    vendor_csv = export_root.with_name(f"{export_root.name}_ddG.csv")
    if not vendor_csv.is_file():
        raise RuntimeError(f"fmp2excel.py did not create expected file: {vendor_csv}")
    if _sha256(fmp) != source_fmp_sha256:
        raise RuntimeError("source FMP changed while fmp2excel.py was running")
    return vendor_csv


def _read_completed_edges(
    vendor_csv: Path,
    *,
    state: str,
    expected_pairs: frozenset[tuple[str, str]],
) -> ParsedEdgeFile:
    """Read raw Bennett estimates and require the immutable input edge set."""
    parsed = read_edge_file(vendor_csv, expected_state=state)
    observed_pairs = frozenset((edge.source, edge.target) for edge in parsed.edges)
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            "completed FMP edge set does not match its input-map snapshot; "
            f"missing={missing}, extra={extra}"
        )
    return parsed


def _extract_completed_mapping_fingerprint(
    fmp: Path,
    *,
    runner: Path,
    output: Path,
) -> str:
    """Extract and validate the actual mapping fingerprint in a completed FMP."""
    script = Path(__file__).with_name("extract_edge_mappings.py")
    command = [
        str(runner),
        "python3",
        str(script),
        "-f",
        str(fmp),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "completed-FMP atom-mapping extraction failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    if not output.is_file():
        raise RuntimeError(f"atom-mapping extractor did not create expected file: {output}")
    try:
        payload = json.loads(output.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{output}: invalid completed-FMP mapping JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError(f"{output}: expected mapping schema_version 2")
    fingerprint = payload.get("mapping_fingerprint")
    if not isinstance(fingerprint, str) or not _is_sha256(fingerprint):
        raise ValueError(f"{output}: mapping_fingerprint must be a lowercase SHA-256")
    return fingerprint


def extract(
    fmp: Path,
    output: Path,
    *,
    state: str,
    schrodinger: Path,
    manifest: Path,
) -> int:
    """Export and normalize raw FEP+ Bennett edge results.

    Args:
        fmp: Completed FEP+ output map.
        output: Destination normalized CSV.
        state: State label, ``open`` or ``closed``.
        schrodinger: Schrodinger Suite installation root.
        manifest: Launch manifest associated with the completed run.

    Returns:
        Number of normalized perturbation edges written.

    Raises:
        FileNotFoundError: If the FMP or Suite runner is missing.
        RuntimeError: If the vendor exporter fails.
        ValueError: If the vendor output has missing or invalid results.
    """
    runner = _validate_extract_request(
        fmp,
        output,
        state=state,
        schrodinger=schrodinger,
        manifest=manifest,
    )
    run = _read_manifest(manifest, state=state, fmp=fmp, schrodinger=schrodinger)
    source_fmp_sha256 = _sha256(fmp)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fep-export-", dir=output.parent) as tmp_name:
        tmp_dir = Path(tmp_name)
        export_root = tmp_dir / "vendor"
        vendor_csv = _export_vendor_table(
            fmp,
            runner=runner,
            export_root=export_root,
            source_fmp_sha256=source_fmp_sha256,
        )
        completed_mapping_fingerprint = _extract_completed_mapping_fingerprint(
            fmp,
            runner=runner,
            output=tmp_dir / "completed_mappings.json",
        )
        if completed_mapping_fingerprint != run.expected_mapping_fingerprint:
            raise ValueError(
                "completed FMP atom mapping does not match its immutable input snapshot: "
                f"{completed_mapping_fingerprint} != {run.expected_mapping_fingerprint}"
            )
        parsed = _read_completed_edges(
            vendor_csv,
            state=state,
            expected_pairs=run.expected_edge_pairs,
        )
        vendor_edge_table_sha256 = _sha256(vendor_csv)
        qc_by_pair = _read_vendor_qc(vendor_csv)
        normalized_tmp = tmp_dir / "normalized.csv"
        with normalized_tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=NORMALIZED_FIELDS)
            writer.writeheader()
            for edge_index, edge in enumerate(parsed.edges):
                writer.writerow({
                    "schema_version": 2,
                    "engine": "fep_plus",
                    "protocol_fingerprint": run.protocol_fingerprint,
                    "protocol_json": run.protocol_json,
                    "run_manifest_sha256": run.manifest_sha256,
                    "run_id": run.run_id,
                    "run_seed": run.seed,
                    "source_fmp_sha256": source_fmp_sha256,
                    **run.input_lineage,
                    "vendor_edge_table_sha256": vendor_edge_table_sha256,
                    **qc_by_pair[(edge.source, edge.target)],
                    "state": state,
                    "edge_id": f"e{edge_index:04d}",
                    "source": edge.source,
                    "target": edge.target,
                    "ddg": f"{edge.ddg:.17g}",
                    "standard_uncertainty": f"{edge.uncertainty:.17g}",
                    "unit": "kcal/mol",
                    "sign_convention": "target_minus_source",
                })
        read_edge_file(
            normalized_tmp,
            expected_state=state,
            require_commit_marker=False,
        )
        vendor_exports = {
            "edges": vendor_csv,
            "nodes": export_root.with_name(f"{export_root.name}_dG.csv"),
            "summary": export_root.with_name(f"{export_root.name}_summary.csv"),
            "hysteresis": export_root.with_name(f"{export_root.name}_hysteresis.csv"),
        }
        final_vendor_exports = {
            label: output.with_name(f"{output.stem}.vendor_{label}.csv") for label in vendor_exports
        }
        provenance = {
            "schema_version": 2,
            "publication_status": "complete_commit_marker",
            "normalized_result": str(output.resolve()),
            "normalized_result_sha256": _sha256(normalized_tmp),
            "source_fmp": str(fmp.resolve()),
            "source_fmp_sha256": source_fmp_sha256,
            "run_manifest": str(manifest.resolve()),
            "run_manifest_sha256": run.manifest_sha256,
            "run_id": run.run_id,
            "run_seed": run.seed,
            "state": state,
            "suite_release": run.suite_release,
            "scientific_protocol": run.protocol,
            "protocol_fingerprint": run.protocol_fingerprint,
            "input_lineage": run.input_lineage,
            "completed_atom_mapping_fingerprint": completed_mapping_fingerprint,
            "adapter": "fmp2excel.py -csv -cycle-closure",
            "statistical_fields": ["bennett_ddg", "bennett_ddg_error"],
            "ignored_as_statistical_inputs": ["pred_dg", "ccc_ddg", "ccc_ddg_error"],
            "vendor_exports": {
                label: {
                    "path": str(final_vendor_exports[label].resolve()),
                    "sha256": _sha256(path),
                }
                for label, path in sorted(vendor_exports.items())
                if path.is_file()
            },
        }
        provenance_tmp = tmp_dir / "provenance.json"
        provenance_tmp.write_text(json.dumps(provenance, indent=2) + "\n")
        provenance_path = output.with_suffix(f"{output.suffix}.provenance.json")
        lock_path = output.with_suffix(f"{output.suffix}.lock")
        if _sha256(fmp) != source_fmp_sha256:
            raise RuntimeError("source FMP changed before result publication")
        with lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if _sha256(fmp) != source_fmp_sha256:
                raise RuntimeError("source FMP changed while waiting to publish results")
            # The provenance file is the commit marker. Its absence means a
            # publication was interrupted and the sibling bundle is incomplete.
            provenance_path.unlink(missing_ok=True)
            for label, vendor_path in vendor_exports.items():
                published_path = final_vendor_exports[label]
                if vendor_path.is_file():
                    os.replace(vendor_path, published_path)
                else:
                    published_path.unlink(missing_ok=True)
            os.replace(normalized_tmp, output)
            os.replace(provenance_tmp, provenance_path)
        return len(parsed.edges)


def main() -> None:
    """Run the command-line extractor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fmp", type=Path, help="Completed FEP+ *_out.fmp file.")
    parser.add_argument("--state", choices=("open", "closed"), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Run manifest (default: manifest.tsv beside the FMP).",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--schrodinger",
        type=Path,
        default=Path(os.environ.get("SCHRODINGER", "/apps/schrodinger2025-4")),
        help="Schrodinger Suite root (default: $SCHRODINGER).",
    )
    args = parser.parse_args()
    manifest = args.manifest if args.manifest is not None else args.fmp.parent / "manifest.tsv"
    try:
        count = extract(
            args.fmp,
            args.output,
            state=args.state,
            schrodinger=args.schrodinger,
            manifest=manifest,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {count} raw Bennett edges to {args.output}")


if __name__ == "__main__":
    main()
