"""Single-receptor-state FEP+ relative binding free-energy analysis.

Reference-independent contrasts retain the full network covariance. A zero
reference value fixes the gauge and is not an absolute binding measurement.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mdpp._types import StrPath
from mdpp.analysis.fepp import (
    _json_default,
    _publish_analysis_bundle,
    _write_diagnostic_tables,
    _write_rows,
    combine_repeats,
    compute_fep_network,
)
from mdpp.core.fepp_results import SUPPORTED_UNITS, ParsedEdgeFile


@dataclass(frozen=True, slots=True)
class RBFEResult:
    """Single-state RBFE summary and covariance-aware diagnostic tables.

    ``summary`` and ``tables`` preserve the existing standalone result schema.
    Summary energies and pairwise contrasts use ``summary["unit"]``; residual
    and cycle tables retain named kcal/mol fields until serialization.
    Frozen fields prevent reassignment; nested containers remain mutable.
    """

    summary: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]


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


def compute_rbfe(
    paths: Sequence[StrPath],
    *,
    state: str,
    reference: str,
    output_unit: str = "kcal/mol",
) -> RBFEResult:
    """Analyze independent repeats of one receptor-state RBFE network.

    Args:
        paths: One normalized result CSV per independent repeat.
        state: Receptor-state label, ``open`` or ``closed``.
        reference: Ligand defining the relative-free-energy gauge.
        output_unit: ``kcal/mol`` or ``kJ/mol``.

    Returns:
        Frozen result containing a JSON-serializable ``summary`` and detailed
        pairwise, residual, and cycle diagnostic ``tables``.

    Raises:
        ValueError: If inputs, provenance, state, or units are invalid.
    """
    if state not in {"open", "closed"}:
        raise ValueError(f"state must be 'open' or 'closed', got {state!r}")
    if output_unit not in SUPPORTED_UNITS:
        raise ValueError(f"unsupported output unit {output_unit!r}")
    edges, files = combine_repeats(paths, state=state)
    _require_normalized(files)
    fit = compute_fep_network(edges, reference=reference)
    protocol_json = files[0].protocol_json
    if protocol_json is None:
        raise ValueError(f"{files[0].path}: normalized scientific protocol is missing")
    factor = 1.0 / SUPPORTED_UNITS[output_unit]
    estimate = fit.relative_free_energy_kcal_mol
    covariance = fit.covariance_kcal2_mol2
    standard_uncertainty = fit.standard_uncertainty_kcal_mol
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
        "ligands": fit.ligands,
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
        "diagnostics": fit.diagnostics,
    }
    tables = {
        "pairwise_contrasts": _pairwise_contrasts(
            fit.ligands,
            estimate,
            covariance,
            factor=factor,
            unit=output_unit,
            state=state,
        ),
        "edge_residuals": fit.edge_residuals,
        "cycles": fit.cycles,
    }
    return RBFEResult(summary=analysis, tables=tables)


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

    _write_diagnostic_tables(
        {name: tables[name] for name in ("edge_residuals", "cycles")},
        output_dir=output_dir,
        unit=analysis["unit"],
        state=analysis["state"],
    )


def write_rbfe(
    result: RBFEResult,
    *,
    output_dir: StrPath,
) -> None:
    """Stage and publish a locked standalone RBFE bundle with a commit marker.

    Args:
        result: Output from :func:`compute_rbfe`.
        output_dir: Destination directory.

    Returns:
        None.

    Raises:
        ValueError: If the staged output bundle is incomplete.
        OSError: If files cannot be written or published.
    """
    analysis, tables = result.summary, result.tables
    output_dir = Path(output_dir)
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
    _publish_analysis_bundle(
        analysis,
        tables,
        output_dir=output_dir,
        expected_names=expected_names,
        stale_names=stale_paired_names,
        write_files=_write_output_files,
        state=analysis["state"],
    )
