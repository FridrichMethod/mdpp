"""Ligand parameterization and topology assignment utilities."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem


def assign_topology(mol: Chem.Mol, template_mol: Chem.Mol) -> Chem.Mol:
    """Assign bond orders and hydrogens from a template molecule to a ligand.

    Uses the template (typically from SMILES) without hydrogens to assign
    bond orders to the ligand's heavy-atom coordinates, then adds hydrogens
    with 3D coordinates.

    Args:
        mol: The ligand molecule (usually from a PDB/MOL2 with no bond orders).
        template_mol: The reference molecule with correct bond orders.

    Returns:
        A new molecule with assigned bond orders and added hydrogens.
    """
    template = Chem.RemoveHs(template_mol)
    Chem.SanitizeMol(template)
    mol_templated = AllChem.AssignBondOrdersFromTemplate(template, mol)
    mol_fixed = Chem.AddHs(mol_templated, addCoords=True)
    return mol_fixed


def constraint_minimization(mol: Chem.Mol, *, max_iters: int = 5000) -> Chem.Mol:
    """Minimize hydrogen positions while keeping heavy atoms fixed.

    Uses the Universal Force Field (UFF) with fixed-point constraints on
    all non-hydrogen atoms.

    Args:
        mol: Input molecule with 3D coordinates (conformer 0).
        max_iters: Maximum number of minimization iterations.

    Returns:
        The molecule with optimized hydrogen positions.
    """
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            ff.AddFixedPoint(atom.GetIdx())
    ff.Minimize(maxIts=max_iters)
    return mol
