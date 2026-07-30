#!/usr/bin/env python3
"""Validate FEP+ ligand SDF chemistry against the SMILES provenance table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rdkit import Chem


def read_templates(path: Path) -> dict[str, Chem.Mol]:
    """Read the two-column ``smiles title`` provenance file.

    Args:
        path: SMILES provenance file with a header row.

    Returns:
        Mapping from unique ligand title to parsed RDKit molecule.

    Raises:
        ValueError: If a row is malformed, duplicated, or chemically invalid.
    """
    templates: dict[str, Chem.Mol] = {}
    lines = path.read_text().splitlines()
    if not lines or lines[0].split() != ["smiles", "title"]:
        raise ValueError(f"{path}: expected header 'smiles title'")
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}: line {line_number}: expected SMILES and title")
        smiles, title = fields
        if title in templates:
            raise ValueError(f"{path}: duplicate ligand title {title!r}")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"{path}: line {line_number}: invalid SMILES for {title}")
        templates[title] = molecule
    if not templates:
        raise ValueError(f"{path}: no ligand templates")
    return templates


def normalized_isomeric_smiles(molecule: Chem.Mol, *, infer_stereo_3d: bool) -> str:
    """Return representation-normalized isomeric SMILES.

    The input SDFs contain reliable 3D coordinates but V2000 can encode a
    double bond as ``either``.  Reassigning double-bond stereo from 3D recovers
    the ACCO Z geometry.  Tetrahedral phosphorus tags are cleared because the
    two equivalent terminal oxygens make that tag representation-dependent.

    Args:
        molecule: Molecule to normalize without modifying the caller's object.
        infer_stereo_3d: Whether to reconstruct stereochemistry from 3D.

    Returns:
        Canonical isomeric SMILES after removing explicit hydrogens.
    """
    normalized = Chem.Mol(molecule)
    for atom in normalized.GetAtoms():
        if atom.GetAtomicNum() == 15:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if infer_stereo_3d:
        for bond in normalized.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                bond.SetBondDir(Chem.BondDir.NONE)
        Chem.AssignStereochemistryFrom3D(normalized, replaceExistingTags=True)
        for atom in normalized.GetAtoms():
            if atom.GetAtomicNum() == 15:
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    normalized = Chem.RemoveHs(normalized)
    Chem.SanitizeMol(normalized)
    Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
    return Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True)


def validate(
    smiles_path: Path,
    ligand_dir: Path,
    *,
    expected_charge: int,
) -> dict[str, Any]:
    """Validate titles, connectivity, stereochemistry, and formal charges.

    Args:
        smiles_path: SMILES provenance table.
        ligand_dir: Directory containing one ``<title>.sdf`` per template.
        expected_charge: Required net formal charge for every ligand.

    Returns:
        JSON-serializable validation report.

    Raises:
        ValueError: If any ligand or set-level invariant fails.
    """
    templates = read_templates(smiles_path)
    sdf_paths = sorted(ligand_dir.glob("*.sdf"))
    actual_names = {path.stem for path in sdf_paths}
    expected_names = set(templates)
    errors: list[str] = []
    if actual_names != expected_names:
        errors.append(
            "ligand file set differs from provenance: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    records: list[dict[str, Any]] = []
    for sdf_path in sdf_paths:
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
        if len(supplier) != 1:
            errors.append(f"{sdf_path}: expected exactly one SDF record, got {len(supplier)}")
            continue
        molecule: Chem.Mol | None = supplier[0]
        if molecule is None:
            errors.append(f"{sdf_path}: molecule could not be parsed and sanitized")
            continue
        if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
            errors.append(f"{sdf_path}: expected exactly one 3D conformer")
            continue
        title = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
        if title != sdf_path.stem:
            errors.append(
                f"{sdf_path}: SDF title {title!r} does not match filename {sdf_path.stem!r}"
            )
        if sdf_path.stem not in templates:
            continue
        charge = Chem.GetFormalCharge(molecule)
        if charge != expected_charge:
            errors.append(
                f"{sdf_path}: formal charge {charge} does not match expected {expected_charge}"
            )
        actual_smiles = normalized_isomeric_smiles(molecule, infer_stereo_3d=True)
        expected_smiles = normalized_isomeric_smiles(
            templates[sdf_path.stem], infer_stereo_3d=False
        )
        if actual_smiles != expected_smiles:
            errors.append(
                f"{sdf_path}: chemistry differs from provenance\n"
                f"  expected: {expected_smiles}\n  observed: {actual_smiles}"
            )
        records.append({
            "title": sdf_path.stem,
            "formal_charge": charge,
            "canonical_isomeric_smiles": actual_smiles,
        })
    if errors:
        raise ValueError("ligand validation failed:\n- " + "\n- ".join(errors))
    return {
        "schema_version": 1,
        "status": "valid",
        "expected_formal_charge": expected_charge,
        "ligand_count": len(records),
        "ligands": records,
    }


def main() -> None:
    """Run the command-line validator."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smiles",
        type=Path,
        default=here / "inputs" / "ligands_amp_fep.smi",
    )
    parser.add_argument(
        "--ligand-dir",
        type=Path,
        default=here / "inputs" / "ligands",
    )
    parser.add_argument("--expected-charge", type=int, default=-1)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(
            args.smiles,
            args.ligand_dir,
            expected_charge=args.expected_charge,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"validated {report['ligand_count']} ligand inputs")


if __name__ == "__main__":
    main()
