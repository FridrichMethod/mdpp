"""Molecule structure drawing utilities using RDKit."""

from __future__ import annotations

from collections.abc import Sequence
from warnings import warn

from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def draw_mol(
    mol: Chem.rdchem.Mol,
    *,
    pattern: Chem.rdchem.Mol | None = None,
    img_size: tuple[int, int] = (300, 300),
    highlight: bool = True,
    alpha: float = 0.5,
) -> Image.Image:
    """Draw a single molecule as a PIL image.

    Args:
        mol: An RDKit molecule.
        pattern: SMARTS pattern to align and highlight as a substructure.
        img_size: Image dimensions in pixels ``(width, height)``.
        highlight: Highlight matched substructure atoms.
        alpha: Transparency of the highlight colour.

    Returns:
        A PIL image.
    """
    Chem.rdDepictor.SetPreferCoordGen(True)

    highlight_atoms: list[int] = []
    highlight_bonds: list[int] = []

    if pattern is not None:
        AllChem.Compute2DCoords(pattern)
        if mol.HasSubstructMatch(pattern):
            AllChem.GenerateDepictionMatching2DStructure(mol, pattern)
            highlight_atoms = list(mol.GetSubstructMatch(pattern))
            highlight_bonds = [
                mol.GetBondBetweenAtoms(
                    highlight_atoms[bond.GetBeginAtomIdx()],
                    highlight_atoms[bond.GetEndAtomIdx()],
                ).GetIdx()
                for bond in pattern.GetBonds()
            ]
        else:
            warn("The pattern is not found in the molecule.", stacklevel=2)

    return Draw.MolToImage(
        mol,
        size=img_size,
        highlightAtoms=highlight_atoms if highlight else [],
        highlightBonds=highlight_bonds,
        highlightColor=(1, 0, 0, alpha),
    )


def draw_mols(
    mols: Sequence[Chem.rdchem.Mol],
    *,
    output_file: str | None = None,
    legends: Sequence[str] | None = None,
    pattern: Chem.rdchem.Mol | None = None,
    mols_per_row: int = 8,
    sub_img_size: tuple[int, int] = (300, 300),
    highlight: bool = True,
) -> Image.Image | None:
    """Draw a grid of molecules as a PIL image.

    Args:
        mols: RDKit molecules to draw.
        output_file: If given, save the image to this path.
        legends: Per-molecule labels shown beneath each cell.
        pattern: SMARTS pattern to align and highlight as a substructure.
        mols_per_row: Number of molecules per grid row.
        sub_img_size: Size of each cell in pixels ``(width, height)``.
        highlight: Highlight matched substructure atoms.

    Returns:
        A PIL image, or None if drawing fails due to layout constraints.
    """
    Chem.rdDepictor.SetPreferCoordGen(True)

    highlight_atom_lists: list[list[int]] | None = None
    highlight_bond_lists: list[list[int]] | None = None

    if pattern is not None:
        AllChem.Compute2DCoords(pattern)
        highlight_atom_lists = []
        highlight_bond_lists = []
        for mol in mols:
            if mol.HasSubstructMatch(pattern):
                AllChem.GenerateDepictionMatching2DStructure(mol, pattern)
                atoms = list(mol.GetSubstructMatch(pattern))
                highlight_atom_lists.append(atoms)
                highlight_bond_lists.append([
                    mol.GetBondBetweenAtoms(
                        atoms[bond.GetBeginAtomIdx()],
                        atoms[bond.GetEndAtomIdx()],
                    ).GetIdx()
                    for bond in pattern.GetBonds()
                ])
            else:
                highlight_atom_lists.append([])
                highlight_bond_lists.append([])

    try:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=legends,
            highlightAtomLists=highlight_atom_lists if highlight else None,
            highlightBondLists=highlight_bond_lists,
        )
    except RuntimeError:
        warn(
            "molsPerRow is too small to draw a large number of molecules, try to increase it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if output_file:
        img.save(output_file)

    return img
