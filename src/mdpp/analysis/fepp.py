"""Paired-state FEP+ selectivity and relative free-energy network estimators.

Only raw directed Bennett edges enter the estimator. Independent repeats use
inverse-variance weighting and edgewise Birge inflation; network fits retain
the full covariance in the reference-ligand gauge. Energies are computed in
kcal/mol and converted only in the serializable output summaries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from mdpp._types import StrPath
from mdpp.core.fepp_results import (
    ANCHOR_QUANTITIES,
    SUPPORTED_UNITS,
    Edge,
    ParsedEdgeFile,
    read_edge_file,
)


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


@dataclass(frozen=True, slots=True)
class FEPNetworkResult:
    """Reference-gauged network fit with full covariance in kcal/mol.

    Array order follows ``ligands``; ``covariance_kcal2_mol2`` has shape
    ``(n_ligands, n_ligands)``. The reference row and column are exactly zero.
    """

    ligands: list[str]
    reference: str
    relative_free_energy_kcal_mol: NDArray[np.float64]
    covariance_kcal2_mol2: NDArray[np.float64]
    standard_uncertainty_kcal_mol: NDArray[np.float64]
    diagnostics: dict[str, Any]
    edge_residuals: list[dict[str, Any]]
    cycles: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class FEPSelectivityResult:
    """Paired-state selectivity summary and detailed diagnostic tables.

    ``summary`` preserves the normalized analysis JSON schema, including the
    full state and selectivity covariance matrices in its declared ``unit``.
    ``tables`` retains explicitly named kcal/mol fields until serialization.
    Frozen fields prevent reassignment; nested containers remain mutable.
    """

    summary: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]


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
                field
                for field in baseline.keys() | current.keys()
                if baseline.get(field) != current.get(field)
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
    paths: Sequence[StrPath],
    *,
    state: str,
) -> tuple[list[CombinedEdge], list[ParsedEdgeFile]]:
    """Combine independent repeats with inverse-variance weights and Birge inflation.

    Args:
        paths: Matching edge-result CSVs, one per independent simulation repeat.
        state: Expected receptor-state label in every result file.

    Returns:
        Combined edges in kcal/mol and their validated input-file records.

    Raises:
        ValueError: If inputs are empty, invalid, incompatible, or not independent.
        OSError: If an input or publication marker cannot be read.
    """
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


def compute_fep_network(
    edges: list[CombinedEdge],
    *,
    reference: str,
) -> FEPNetworkResult:
    """Fit a connected RBFE network using diagonal-error weighted least squares.

    Args:
        edges: Canonically oriented combined edges in kcal/mol with positive
            standard uncertainties. Edge errors are assumed independent.
        reference: Ligand defining the reference value and covariance zero.

    Returns:
        A reference-gauged fit, full covariance, residuals, and cycle diagnostics.

    Raises:
        ValueError: If the reference is absent or the network is rank deficient.
    """
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

    return FEPNetworkResult(
        ligands=ligands,
        reference=reference,
        relative_free_energy_kcal_mol=estimate,
        covariance_kcal2_mol2=covariance,
        standard_uncertainty_kcal_mol=np.sqrt(np.diag(covariance)),
        diagnostics={
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
        edge_residuals=residuals,
        cycles=_cycle_diagnostics(edges, ligands=ligands, reference=reference),
    )


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


def _converted_fit(fit: FEPNetworkResult, *, factor: float) -> dict[str, Any]:
    """Convert a fit dictionary from internal kcal/mol to the output unit."""
    return {
        "relative_free_energy": (fit.relative_free_energy_kcal_mol * factor).tolist(),
        "standard_uncertainty": (fit.standard_uncertainty_kcal_mol * factor).tolist(),
        "covariance": (fit.covariance_kcal2_mol2 * factor**2).tolist(),
        "diagnostics": fit.diagnostics,
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


def compute_fep_selectivity(
    open_paths: Sequence[StrPath],
    *,
    closed_paths: Sequence[StrPath],
    reference: str,
    output_unit: str = "kcal/mol",
    anchor_path: StrPath | None = None,
) -> FEPSelectivityResult:
    """Analyze paired open/closed result networks.

    Args:
        open_paths: One normalized schema-v2 CSV per independent open repeat.
        closed_paths: One normalized schema-v2 CSV per independent closed repeat.
        reference: Ligand defining the relative-free-energy gauge.
        output_unit: ``kcal/mol`` or ``kJ/mol``.
        anchor_path: Optional independent reference anchor JSON.

    Returns:
        Frozen result with a JSON-serializable ``summary`` and detailed edge
        and cycle diagnostic ``tables``. An independent anchor adds a common
        rank-one covariance term to the paired double differences.

    Raises:
        ValueError: If inputs are invalid or the two networks are incomparable.
    """
    if output_unit not in SUPPORTED_UNITS:
        raise ValueError(f"unsupported output unit {output_unit!r}")
    open_edges, open_files = combine_repeats(open_paths, state="open")
    closed_edges, closed_files = combine_repeats(closed_paths, state="closed")
    all_files = _validate_paired_inputs(open_files, closed_files)
    open_fit = compute_fep_network(open_edges, reference=reference)
    closed_fit = compute_fep_network(closed_edges, reference=reference)
    if open_fit.ligands != closed_fit.ligands:
        open_only = sorted(set(open_fit.ligands) - set(closed_fit.ligands))
        closed_only = sorted(set(closed_fit.ligands) - set(open_fit.ligands))
        raise ValueError(
            f"open/closed ligand sets differ: open_only={open_only}, closed_only={closed_only}"
        )

    ligands: list[str] = open_fit.ligands
    relative = open_fit.relative_free_energy_kcal_mol - closed_fit.relative_free_energy_kcal_mol
    relative_covariance = open_fit.covariance_kcal2_mol2 + closed_fit.covariance_kcal2_mol2
    factor = 1.0 / SUPPORTED_UNITS[output_unit]
    anchored: dict[str, Any] | None = None
    if anchor_path is not None:
        anchor = _read_anchor(Path(anchor_path), reference=reference)
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
        "open_edge_residuals": open_fit.edge_residuals,
        "closed_edge_residuals": closed_fit.edge_residuals,
        "open_cycles": open_fit.cycles,
        "closed_cycles": closed_fit.cycles,
    }
    return FEPSelectivityResult(summary=analysis, tables=tables)


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

    _write_diagnostic_tables(tables, output_dir=output_dir, unit=unit)


def _write_diagnostic_tables(
    tables: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
    unit: str,
    state: str | None = None,
) -> None:
    """Convert kcal/mol diagnostics and serialize nested cycle fields as CSV rows."""
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
            if state is not None:
                row["state"] = state
            row["unit"] = unit
            converted_rows.append(row)
        _write_rows(output_dir / f"{name}.csv", converted_rows)


def write_fep_selectivity(
    result: FEPSelectivityResult,
    *,
    output_dir: StrPath,
) -> None:
    """Publish a paired-state analysis bundle under an exclusive file lock.

    Args:
        result: Output from :func:`compute_fep_selectivity`.
        output_dir: Directory for analysis JSON, diagnostic CSVs, and commit marker.

    Returns:
        None.

    Raises:
        ValueError: If the staged bundle is incomplete or not JSON serializable.
        OSError: If files cannot be written or published.
    """
    analysis, tables = result.summary, result.tables
    output_dir = Path(output_dir)
    expected_names = {
        "analysis.json",
        "selectivity.csv",
        "open_edge_residuals.csv",
        "closed_edge_residuals.csv",
        "open_cycles.csv",
        "closed_cycles.csv",
    }
    stale_single_names = {
        "relative_free_energies.csv",
        "pairwise_contrasts.csv",
        "edge_residuals.csv",
        "cycles.csv",
    }
    _publish_analysis_bundle(
        analysis,
        tables,
        output_dir=output_dir,
        expected_names=expected_names,
        stale_names=stale_single_names,
        write_files=_write_output_files,
    )


def _publish_analysis_bundle(
    analysis: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
    expected_names: set[str],
    stale_names: set[str],
    write_files: Callable[..., None],
    state: str | None = None,
) -> None:
    """Stage and lock-publish matching analysis files with their final commit marker."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f"{output_dir.name}.lock")
    marker_name = "analysis.provenance.json"
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ) as staging_name:
        staging_dir = Path(staging_name)
        write_files(analysis, tables, output_dir=staging_dir)
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
        if state is not None:
            marker["state"] = state
        staged_marker = staging_dir / marker_name
        staged_marker.write_text(json.dumps(marker, indent=2) + "\n")

        with lock_path.open("a+") as lock_handle:
            # File publication uses POSIX locks; importing mdpp does not.
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            output_dir.mkdir(parents=True, exist_ok=True)
            published_marker = output_dir / marker_name
            published_marker.unlink(missing_ok=True)
            for name in sorted(stale_names):
                (output_dir / name).unlink(missing_ok=True)
            for name in sorted(expected_names):
                os.replace(staging_dir / name, output_dir / name)
            os.replace(staged_marker, published_marker)
