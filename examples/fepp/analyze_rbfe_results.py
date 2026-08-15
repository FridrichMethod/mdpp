#!/usr/bin/env python3
"""Analyze one-conformation, multi-ligand FEP+ RBFE networks.

With reference ligand ``r``, this script estimates the identifiable quantity

    DeltaDeltaG_i,r = DeltaG_bind(i) - DeltaG_bind(r).

Negative values mean stronger predicted binding than the reference under the
usual binding-free-energy sign convention. The reference value of zero and its
zero uncertainty fix the network gauge; they are not an absolute binding-free-
energy measurement. Production analysis requires normalized, provenance-
bearing exports written by ``extract_fep_results.py``.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from analyze_fep_results import SUPPORTED_UNITS, ParsedEdgeFile, combine_repeats, fit_network


def _require_normalized(files: list[ParsedEdgeFile]) -> None:
    """Require complete normalized provenance for a production result."""
    for result in files:
        if result.schema not in {"normalized_v2", "normalized_v3"} or any(
            value is None
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
                result.workflow_type,
            )
        ):
            raise ValueError(
                "single-state analysis requires normalized exports from "
                "extract_fep_results.py with run-manifest provenance"
            )


def _provenance(files: list[ParsedEdgeFile]) -> list[dict[str, Any]]:
    """Return serialized provenance records for normalized result files."""
    return [
        {
            "path": str(result.path),
            "sha256": result.sha256,
            "schema": result.schema,
            "workflow_type": result.workflow_type,
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


def _quality_control(files: list[ParsedEdgeFile]) -> dict[str, Any]:
    """Aggregate vendor QC summaries without excluding warning edges."""
    runs = []
    warning_count = 0
    for result in files:
        if result.quality_control is None:
            raise ValueError(f"{result.path}: normalized vendor QC is missing")
        warning_count += int(result.quality_control["warning_edge_count"])
        runs.append({"run_id": result.run_id, "summary": result.quality_control})
    return {
        "has_warnings": warning_count > 0,
        "warning_edge_ratings_across_runs": warning_count,
        "runs": runs,
        "policy": "diagnostic_only_no_automatic_edge_exclusion",
    }


def _pairwise_contrasts(
    ligands: list[str],
    estimate: np.ndarray,
    covariance: np.ndarray,
    *,
    factor: float,
    unit: str,
    state: str,
) -> list[dict[str, Any]]:
    """Return every reference-invariant pairwise node contrast."""
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(ligands):
        for target_index in range(source_index + 1, len(ligands)):
            target = ligands[target_index]
            variance = (
                covariance[source_index, source_index]
                + covariance[target_index, target_index]
                - 2.0 * covariance[source_index, target_index]
            )
            if variance < -1e-12:
                raise ValueError("fitted covariance gives a negative pairwise variance")
            rows.append({
                "state": state,
                "source": source,
                "target": target,
                "target_minus_source": (estimate[target_index] - estimate[source_index]) * factor,
                "standard_uncertainty": math.sqrt(max(0.0, variance)) * factor,
                "unit": unit,
            })
    return rows


def analyze(
    paths: list[Path],
    *,
    state: str,
    reference: str,
    output_unit: str = "kcal/mol",
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Analyze independent repeats of one receptor-state RBFE network.

    Args:
        paths: One normalized result CSV per independent repeat.
        state: Receptor-state label, ``open`` or ``closed``.
        reference: Ligand defining the relative-free-energy gauge.
        output_unit: ``kcal/mol`` or ``kJ/mol``.

    Returns:
        ``(analysis, tables)`` with a JSON-serializable summary and detailed
        pairwise, residual, and cycle rows.

    Raises:
        ValueError: If inputs, provenance, state, or units are invalid.
    """
    if state not in {"open", "closed"}:
        raise ValueError(f"state must be 'open' or 'closed', got {state!r}")
    if output_unit not in SUPPORTED_UNITS:
        raise ValueError(f"unsupported output unit {output_unit!r}")
    edges, files = combine_repeats(paths, state=state)
    _require_normalized(files)
    fit = fit_network(edges, reference=reference)
    protocol_json = files[0].protocol_json
    if protocol_json is None:
        raise ValueError(f"{files[0].path}: normalized scientific protocol is missing")
    factor = 1.0 / SUPPORTED_UNITS[output_unit]
    estimate = fit["relative_free_energy_kcal_mol"]
    covariance = fit["covariance_kcal2_mol2"]
    standard_uncertainty = fit["standard_uncertainty_kcal_mol"]
    workflow_types = sorted({str(result.workflow_type) for result in files})
    analysis = {
        "schema_version": 1,
        "quantity": "reference_relative_binding_free_energy",
        "state": state,
        "sign_convention": (
            "ligand_minus_reference; negative means stronger predicted binding than the reference"
        ),
        "reference": reference,
        "unit": output_unit,
        "ligands": fit["ligands"],
        "scientific_protocol": json.loads(protocol_json),
        "protocol_fingerprint": files[0].protocol_fingerprint,
        "source_workflow_types": workflow_types,
        "quality_control": _quality_control(files),
        "assumptions": [
            "raw directed edge ddG equals G(target) - G(source)",
            "reported edge uncertainties are one-standard-uncertainty sampling errors",
            "edge errors are independent within the network (diagonal covariance)",
            "independent repeats use a fixed-effect inverse-variance mean",
            "repeat disagreement inflates each pooled edge uncertainty by the Birge ratio",
            "chi-square p-values are approximate and assume normal calibrated input errors",
            "vendor QC ratings are diagnostic and do not automatically exclude edges",
            "the reference value 0 +/- 0 fixes the gauge and is not an absolute measurement",
            "shared force-field and other systematic errors are not represented",
        ],
        "input_provenance": _provenance(files),
        "relative_binding_free_energy": {
            "value": (estimate * factor).tolist(),
            "standard_uncertainty": (standard_uncertainty * factor).tolist(),
            "covariance": (covariance * factor**2).tolist(),
        },
        "diagnostics": fit["diagnostics"],
    }
    tables = {
        "pairwise_contrasts": _pairwise_contrasts(
            fit["ligands"],
            estimate,
            covariance,
            factor=factor,
            unit=output_unit,
            state=state,
        ),
        "edge_residuals": fit["edge_residuals"],
        "cycles": fit["cycles"],
    }
    return analysis, tables


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat dictionaries to CSV, or an empty file for no rows."""
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar values to JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_output_files(
    analysis: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
) -> None:
    """Write one unpublished standalone RBFE analysis bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False, default=_json_default) + "\n"
    )
    relative = analysis["relative_binding_free_energy"]
    relative_rows = [
        {
            "state": analysis["state"],
            "ligand": ligand,
            "reference": analysis["reference"],
            "ligand_minus_reference": relative["value"][index],
            "standard_uncertainty": relative["standard_uncertainty"][index],
            "unit": analysis["unit"],
        }
        for index, ligand in enumerate(analysis["ligands"])
    ]
    _write_rows(output_dir / "relative_free_energies.csv", relative_rows)
    _write_rows(output_dir / "pairwise_contrasts.csv", tables["pairwise_contrasts"])

    factor = 1.0 / SUPPORTED_UNITS[analysis["unit"]]
    for name in ("edge_residuals", "cycles"):
        converted_rows: list[dict[str, Any]] = []
        for original in tables[name]:
            row = dict(original)
            for key in list(row):
                if key.endswith("_kcal_mol"):
                    row[key.removesuffix("_kcal_mol")] = row.pop(key) * factor
            if "terms" in row:
                row["terms"] = json.dumps(row["terms"], separators=(",", ":"))
            if "nodes" in row:
                row["nodes"] = " -> ".join([*row["nodes"], row["nodes"][0]])
            row["state"] = analysis["state"]
            row["unit"] = analysis["unit"]
            converted_rows.append(row)
        _write_rows(output_dir / f"{name}.csv", converted_rows)


def write_outputs(
    analysis: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
) -> None:
    """Stage and atomically publish a locked standalone RBFE bundle.

    Args:
        analysis: JSON-serializable result returned by :func:`analyze`.
        tables: Detailed tables returned by :func:`analyze`.
        output_dir: Destination directory.

    Raises:
        ValueError: If the staged output bundle is incomplete.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f"{output_dir.name}.lock")
    marker_name = "analysis.provenance.json"
    expected_names = {
        "analysis.json",
        "relative_free_energies.csv",
        "pairwise_contrasts.csv",
        "edge_residuals.csv",
        "cycles.csv",
    }
    stale_paired_names = {
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
            "state": analysis["state"],
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
            for name in sorted(stale_paired_names):
                (output_dir / name).unlink(missing_ok=True)
            for name in sorted(expected_names):
                os.replace(staging_dir / name, output_dir / name)
            os.replace(staged_marker, published_marker)


def main() -> None:
    """Run the standalone RBFE command-line analyzer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Normalized edge CSV; repeat for independent repeats.",
    )
    parser.add_argument("--state", choices=("open", "closed"), required=True)
    parser.add_argument("--reference", required=True, help="Explicit reference ligand title.")
    parser.add_argument(
        "--unit",
        choices=tuple(SUPPORTED_UNITS),
        default="kcal/mol",
        help="Output energy unit (default: kcal/mol).",
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
            args.input,
            state=args.state,
            reference=args.reference,
            output_unit=args.unit,
        )
        write_outputs(analysis, tables, output_dir=args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"wrote FEP+ {args.state} RBFE analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
