"""Utility functions for MD simulation pre- and post-processing."""

import os
from pathlib import Path

from openmm.app import PDBFile
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem

type StrPath = str | os.PathLike[str]


def assign_topology(mol: Chem.Mol, template_mol: Chem.Mol) -> Chem.Mol:
    """Assigns the topology from the template molecule to the ligand.

    Args:
        mol: The ligand to assign the topology to.
        template_mol: The template molecule to use for the assignment.

    Returns:
        The ligand with the assigned topology.

    """
    # Use the template (usually from SMILES) without hydrogens
    # to assign topology to the ligands coordinates (heavy atoms only)
    template = Chem.RemoveHs(template_mol)
    Chem.SanitizeMol(template)
    mol_templated = AllChem.AssignBondOrdersFromTemplate(template, mol)

    # Add hydrogens to the ligands
    mol_fixed = Chem.AddHs(mol_templated, addCoords=True)

    return mol_fixed


def constraint_minimization(mol: Chem.Mol) -> Chem.Mol:
    """Constraint minimization for hydrogen atoms."""
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 1:
            ff.AddFixedPoint(a.GetIdx())
    ff.Minimize(maxIts=5000)

    return mol


def fix_pdb(pdb_path: StrPath, fixed_pdb_path: StrPath, pH: float = 7.0) -> None:
    """Fixes the PDB file."""
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=pH)

    with Path(fixed_pdb_path).open("w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
