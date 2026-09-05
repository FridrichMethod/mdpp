"""Ligand parameterization and topology assignment utilities."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem


def assign_topology(mol: Chem.Mol, template_mol: Chem.Mol) -> Chem.Mol:
    """Assign bond orders and hydrogens from a template molecule to a ligand.

    Uses the template (typically from SMILES) without hydrogens to assign
    bond orders and stereochemistry to the ligand's heavy-atom coordinates,
    then adds hydrogens with 3D coordinates. Specified template stereochemistry
    must agree with an input 3D conformer; coordinates are never inverted to
    force a match. Input heavy-atom ordering is preserved.

    Args:
        mol: The ligand molecule (usually from a PDB/MOL2 with no bond orders).
        template_mol: The reference molecule with correct bond orders.

    Returns:
        A new molecule with template stereochemistry, bond orders, and hydrogens.

    Raises:
        ValueError: If the ligand and template have different heavy-atom graphs,
            or the input 3D geometry conflicts with specified stereochemistry.
    """
    # The template molecule should have no explicit hydrogens
    # else the AssignBondOrdersFromTemplate algorithm will fail.
    refmol = Chem.RemoveHs(template_mol)
    Chem.SanitizeMol(refmol)

    mol_templated = AllChem.AssignBondOrdersFromTemplate(refmol, Chem.RemoveHs(mol))
    if mol_templated.GetNumAtoms() != refmol.GetNumAtoms():
        raise ValueError("Ligand and template must have the same heavy-atom graph")

    # AssignBondOrdersFromTemplate does not transfer double-bond stereo. Infer
    # stereo on a copy to validate the coordinates, then use the template graph
    # so unspecified centers (e.g. phosphate P) are not invented by the 3D fit.
    geometry = Chem.Mol(mol_templated)
    has_3d = geometry.GetNumConformers() > 0 and geometry.GetConformer().Is3D()
    if has_3d:
        Chem.AssignStereochemistryFrom3D(geometry, replaceExistingTags=True)
    match = geometry.GetSubstructMatch(refmol, useChirality=has_3d)
    if not match:
        raise ValueError("Ligand geometry or topology conflicts with template stereochemistry")

    # RenumberAtoms adjusts atom and bond stereo consistently with the atom
    # permutation. Retain the input coordinates and PDB atom identifiers.
    order = sorted(range(len(match)), key=match.__getitem__)
    mol_fixed = Chem.RenumberAtoms(refmol, order)
    mol_fixed.RemoveAllConformers()
    for conformer in mol_templated.GetConformers():
        mol_fixed.AddConformer(Chem.Conformer(conformer), assignId=True)
    for atom in mol_templated.GetAtoms():
        if (info := atom.GetMonomerInfo()) is not None:
            mol_fixed.GetAtomWithIdx(atom.GetIdx()).SetMonomerInfo(info)
    for key in mol.GetPropNames(includePrivate=True, includeComputed=False):
        mol_fixed.SetProp(key, mol.GetProp(key))
    mol_fixed = Chem.AddHs(mol_fixed, addCoords=True)

    return mol_fixed


def constraint_minimization(mol: Chem.Mol, *, max_iters: int = 5000) -> Chem.Mol:
    """Minimize hydrogen positions while keeping heavy atoms fixed.

    Uses the Universal Force Field (UFF) with fixed-point constraints on
    all non-hydrogen atoms.

    Args:
        mol: Input molecule with 3D coordinates (conformer 0).
        max_iters: Maximum number of minimization iterations.

    Returns:
        The molecule with optimized hydrogen positions (modified in place).

    Raises:
        ValueError: If UFF parameters are unavailable for this molecule.
    """
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
    if ff is None:
        raise ValueError("UFF parameters unavailable for this molecule; cannot minimize.")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            ff.AddFixedPoint(atom.GetIdx())
    ff.Minimize(maxIts=max_iters)

    return mol
