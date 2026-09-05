#!/usr/bin/env python3
"""Compare prepared receptors by sequence, charge, and H-atom connectivity.

Run this script with Schrodinger's Python.  Protonation signatures are derived
from formal charges and the heavy atoms to which hydrogens are bonded; residue
labels such as ``HIE``/``HIP`` are reported but are not trusted as the sole
source of truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASE_RESIDUE_NAMES = {
    "ASH": "ASP",
    "CYM": "CYS",
    "CYX": "CYS",
    "GLH": "GLU",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "LYN": "LYS",
}


def _residue_key(residue: Any) -> str:
    """Return a stable chain/residue/insertion-code identifier."""
    insertion = residue.inscode.strip()
    return f"{residue.chain.strip() or '_'}:{residue.resnum}{insertion}"


def _heavy_atom_topology(atom: Any) -> dict[str, Any]:
    """Return one heavy atom's identity and residue-aware heavy neighbors."""
    bonded_heavy_atoms = sorted(
        (
            {
                "residue": _residue_key(bonded.getResidue()),
                "atom": bonded.pdbname.strip(),
                "element": bonded.element,
            }
            for bonded in atom.bonded_atoms
            if bonded.element != "H"
        ),
        key=lambda item: (item["residue"], item["atom"], item["element"]),
    )
    return {
        "atom": atom.pdbname.strip(),
        "element": atom.element,
        "bonded_heavy_atoms": bonded_heavy_atoms,
    }


def summarize(path: Path) -> dict[str, Any]:
    """Summarize the first structure in a prepared receptor file.

    Args:
        path: Prepared Maestro receptor file.

    Returns:
        JSON-serializable atom, charge, sequence, and microstate summary.

    Raises:
        ValueError: If the file contains zero or multiple structures.
    """
    from schrodinger import structure

    structures = list(structure.StructureReader(str(path)))
    if len(structures) != 1:
        raise ValueError(f"{path}: expected one receptor structure, got {len(structures)}")
    receptor = structures[0]
    residues: dict[str, dict[str, Any]] = {}
    for residue in receptor.residue:
        key = _residue_key(residue)
        if key in residues:
            raise ValueError(f"{path}: duplicate residue identifier {key}")
        label = residue.pdbres.strip()
        atoms = list(residue.atom)
        heavy_signatures = []
        heavy_topology = []
        for atom in atoms:
            if atom.element == "H":
                continue
            heavy_topology.append(_heavy_atom_topology(atom))
            attached_hydrogens = sorted(
                bonded.pdbname.strip() for bonded in atom.bonded_atoms if bonded.element == "H"
            )
            heavy_signatures.append({
                "atom": atom.pdbname.strip(),
                "element": atom.element,
                "formal_charge": int(atom.formal_charge),
                "attached_hydrogens": attached_hydrogens,
            })
        residues[key] = {
            "label": label,
            "base_residue": BASE_RESIDUE_NAMES.get(label, label),
            "formal_charge": int(sum(atom.formal_charge for atom in atoms)),
            "heavy_atom_topology": sorted(heavy_topology, key=lambda item: item["atom"]),
            "heavy_atoms": sorted(heavy_signatures, key=lambda item: item["atom"]),
        }
    return {
        "path": str(path),
        "atom_count": len(receptor.atom),
        "formal_charge": int(sum(atom.formal_charge for atom in receptor.atom)),
        "residues": residues,
    }


def compare_receptor_summaries(
    open_summary: dict[str, Any],
    closed_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare receptor chemistry independently of the structure-file reader.

    Args:
        open_summary: Atom, charge, and residue summary of the open receptor.
        closed_summary: Atom, charge, and residue summary of the closed receptor.

    Returns:
        Schema-version-1 comparison report separating sequence, heavy-atom
        connectivity, and protonation differences. Input summaries are included.

    Raises:
        KeyError: If a required summary field is missing.
    """
    open_residues = open_summary["residues"]
    closed_residues = closed_summary["residues"]
    all_keys = sorted(set(open_residues) | set(closed_residues))
    sequence_mismatches = []
    composition_mismatches = []
    microstate_mismatches = []
    for key in all_keys:
        open_residue = open_residues.get(key)
        closed_residue = closed_residues.get(key)
        if open_residue is None or closed_residue is None:
            sequence_mismatches.append({
                "residue": key,
                "open": open_residue,
                "closed": closed_residue,
            })
            continue
        if open_residue["base_residue"] != closed_residue["base_residue"]:
            sequence_mismatches.append({
                "residue": key,
                "open": open_residue,
                "closed": closed_residue,
            })
            continue
        if open_residue["heavy_atom_topology"] != closed_residue["heavy_atom_topology"]:
            composition_mismatches.append({
                "residue": key,
                "open": open_residue,
                "closed": closed_residue,
            })
            continue
        open_signature = {
            "formal_charge": open_residue["formal_charge"],
            "heavy_atoms": open_residue["heavy_atoms"],
        }
        closed_signature = {
            "formal_charge": closed_residue["formal_charge"],
            "heavy_atoms": closed_residue["heavy_atoms"],
        }
        if open_signature != closed_signature:
            microstate_mismatches.append({
                "residue": key,
                "open": open_residue,
                "closed": closed_residue,
            })

    identical = (
        not sequence_mismatches
        and not composition_mismatches
        and not microstate_mismatches
        and open_summary["formal_charge"] == closed_summary["formal_charge"]
    )
    return {
        "schema_version": 1,
        "identical_declared_microstates": identical,
        "open": open_summary,
        "closed": closed_summary,
        "sequence_mismatches": sequence_mismatches,
        "composition_mismatches": composition_mismatches,
        "microstate_mismatches": microstate_mismatches,
        "formal_charge_difference_open_minus_closed": (
            open_summary["formal_charge"] - closed_summary["formal_charge"]
        ),
    }


def enforce_receptor_microstate_policy(
    report: dict[str, Any],
    *,
    allow_microstate_mismatch: bool = False,
) -> None:
    """Reject sequence changes and unauthorized receptor-microstate changes.

    Args:
        report: Comparison report returned by :func:`compare_receptor_summaries`.
        allow_microstate_mismatch: Whether protonation/formal-charge differences
            may be retained explicitly.

    Returns:
        None.

    Raises:
        ValueError: If receptor sequences differ, or if microstates differ
            without the explicit allowance.
    """
    if report["sequence_mismatches"]:
        raise ValueError(
            "prepared receptor sequences differ; sequence mismatches cannot be overridden"
        )
    if report["composition_mismatches"]:
        raise ValueError(
            "prepared receptor heavy-atom composition/connectivity differs; "
            "composition mismatches cannot be overridden"
        )
    if not report["identical_declared_microstates"] and not allow_microstate_mismatch:
        raise ValueError(
            "prepared receptors have different protonation/formal-charge signatures; "
            "use --allow-microstate-mismatch explicitly"
        )


def main() -> None:
    """Run the command-line receptor comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("open", type=Path, nargs="?", help="Prepared open receptor .mae.")
    parser.add_argument("closed", type=Path, nargs="?", help="Prepared closed receptor .mae.")
    parser.add_argument("--summary", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--allow-microstate-mismatch",
        "--allow-mismatch",
        dest="allow_microstate_mismatch",
        action="store_true",
        help=(
            "Record protonation/formal-charge mismatches but return success; "
            "sequence and heavy-atom composition/connectivity mismatches always fail."
        ),
    )
    args = parser.parse_args()
    if args.summary is not None:
        if args.open is not None or args.closed is not None:
            parser.error("--summary does not accept receptor comparison paths")
        report = summarize(args.summary)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    if args.open is None or args.closed is None:
        parser.error("open and closed receptor paths are required")
    try:
        report = compare_receptor_summaries(summarize(args.open), summarize(args.closed))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        enforce_receptor_microstate_policy(
            report,
            allow_microstate_mismatch=args.allow_microstate_mismatch,
        )
    except ValueError as exc:
        parser.error(f"{exc}; see {args.output}")
    status = "identical" if report["identical_declared_microstates"] else "MISMATCH ALLOWED"
    print(f"receptor microstates: {status}; report: {args.output}")


if __name__ == "__main__":
    main()
