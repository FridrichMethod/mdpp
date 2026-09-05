"""Validate named ligand structures against SMILES provenance.

This module remains standalone-compatible with Python 3.11 so the FEP+ example
can use the RDKit bundled with Schrodinger without importing the mdpp package.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from rdkit import Chem

if TYPE_CHECKING:
    from mdpp._types import StrPath


def read_ligand_templates(path: StrPath) -> dict[str, Chem.Mol]:
    """Read the two-column ``smiles title`` provenance file.

    Args:
        path: SMILES provenance file with a header row.

    Returns:
        Mapping from unique ligand title to parsed RDKit molecule.

    Raises:
        ValueError: If a row is malformed, duplicated, or chemically invalid.
    """
    path = Path(path)
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


def normalized_isomeric_smiles(
    molecule: Chem.Mol,
    *,
    infer_stereo_3d: bool = False,
    ignore_phosphorus_stereo: bool = False,
) -> str:
    """Return representation-normalized isomeric SMILES.

    With ``infer_stereo_3d=True``, coordinates take precedence over encoded
    stereochemical tags, including V2000 double bonds marked as ``either``.
    All stereocenters are preserved by default. The phosphorus opt-out is for
    datasets where phosphate tags are representation-dependent; it must not be
    used when phosphorus configuration is part of the declared chemistry.

    Args:
        molecule: Molecule to normalize without modifying the caller's object.
        infer_stereo_3d: Whether to reconstruct stereochemistry from a 3D conformer.
        ignore_phosphorus_stereo: Explicitly discard every tetrahedral phosphorus
            tag; intended only for datasets with nonphysical phosphate tags.

    Returns:
        Canonical isomeric SMILES after removing explicit hydrogens.

    Raises:
        ValueError: If 3D inference is requested without a 3D conformer, or
            sanitization fails.
    """
    normalized = Chem.Mol(molecule)
    if infer_stereo_3d:
        if not normalized.GetNumConformers() or not normalized.GetConformer().Is3D():
            raise ValueError("3D stereochemistry inference requires a 3D conformer")
        for bond in normalized.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                bond.SetBondDir(Chem.BondDir.NONE)
        Chem.AssignStereochemistryFrom3D(normalized, replaceExistingTags=True)
    if ignore_phosphorus_stereo:
        for atom in normalized.GetAtoms():
            if atom.GetAtomicNum() == 15:
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    normalized = Chem.RemoveHs(normalized)
    Chem.SanitizeMol(normalized)
    Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
    return Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True)


def validate_ligand_inputs(
    smiles_path: StrPath,
    ligand_dir: StrPath,
    *,
    expected_charge: int | None = None,
    ignore_phosphorus_stereo: bool = False,
) -> dict[str, Any]:
    """Validate titles, connectivity, stereochemistry, and formal charges.

    Args:
        smiles_path: SMILES provenance table.
        ligand_dir: Directory containing one ``<title>.sdf`` per template.
        expected_charge: Required net formal charge for every ligand, or ``None``
            to require only agreement with each ligand's SMILES template.
        ignore_phosphorus_stereo: Explicitly discard every tetrahedral phosphorus
            tag during comparison. Defaults to preserving phosphorus configuration.

    Returns:
        JSON-serializable validation report.

    Raises:
        ValueError: If any ligand or set-level invariant fails.
    """
    templates = read_ligand_templates(smiles_path)
    ligand_dir = Path(ligand_dir)
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
        if expected_charge is not None and charge != expected_charge:
            errors.append(
                f"{sdf_path}: formal charge {charge} does not match expected {expected_charge}"
            )
        actual_smiles = normalized_isomeric_smiles(
            molecule, infer_stereo_3d=True, ignore_phosphorus_stereo=ignore_phosphorus_stereo
        )
        expected_smiles = normalized_isomeric_smiles(
            templates[sdf_path.stem],
            infer_stereo_3d=False,
            ignore_phosphorus_stereo=ignore_phosphorus_stereo,
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
