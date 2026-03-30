"""Molecule structure drawing utilities using RDKit."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)

_DEFAULT_HIGHLIGHT_COLOR = (0.0, 0.45, 0.85, 0.6)


def build_draw_options() -> rdMolDraw2D.MolDrawOptions:
    """Create ACS 1996-style drawing options for crisp medicinal chemistry figures."""
    options = rdMolDraw2D.MolDrawOptions()
    rdMolDraw2D.SetACS1996Mode(options, 1.5)
    options.useBWAtomPalette()
    options.bondLineWidth = 1.2

    return options


def get_highlight_bonds(
    mol: Chem.rdchem.Mol,
    atoms: Sequence[int],
    pattern: Chem.rdchem.Mol,
) -> list[int]:
    """Return molecule bond indices corresponding to a matched pattern.

    Args:
        mol: Molecule containing the matched substructure.
        atoms: Molecule atom indices matching the pattern atom order.
        pattern: Pattern whose bonds should be mapped onto ``mol``.

    Returns:
        Bond indices in ``mol`` corresponding to bonds in ``pattern``.
    """
    return [
        mol.GetBondBetweenAtoms(
            atoms[bond.GetBeginAtomIdx()],
            atoms[bond.GetEndAtomIdx()],
        ).GetIdx()
        for bond in pattern.GetBonds()
    ]


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
    drawable_mol = Chem.Mol(mol)

    highlight_atoms: list[int] = []
    highlight_bonds: list[int] = []

    if pattern is not None:
        drawable_pattern = Chem.Mol(pattern)
        AllChem.Compute2DCoords(drawable_pattern)
        if drawable_mol.HasSubstructMatch(drawable_pattern):
            AllChem.GenerateDepictionMatching2DStructure(drawable_mol, drawable_pattern)
            highlight_atoms = list(drawable_mol.GetSubstructMatch(drawable_pattern))
            highlight_bonds = get_highlight_bonds(
                drawable_mol,
                highlight_atoms,
                drawable_pattern,
            )
        else:
            logger.warning("The pattern is not found in the molecule.")

    return Draw.MolToImage(
        drawable_mol,
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
    drawable_mols = [Chem.Mol(mol) for mol in mols]

    highlight_atom_lists: list[list[int]] | None = None
    highlight_bond_lists: list[list[int]] | None = None

    if pattern is not None:
        drawable_pattern = Chem.Mol(pattern)
        AllChem.Compute2DCoords(drawable_pattern)
        highlight_atom_lists = []
        highlight_bond_lists = []
        for drawable_mol in drawable_mols:
            if drawable_mol.HasSubstructMatch(drawable_pattern):
                AllChem.GenerateDepictionMatching2DStructure(drawable_mol, drawable_pattern)
                atoms = list(drawable_mol.GetSubstructMatch(drawable_pattern))
                highlight_atom_lists.append(atoms)
                highlight_bond_lists.append(
                    get_highlight_bonds(drawable_mol, atoms, drawable_pattern)
                )
            else:
                highlight_atom_lists.append([])
                highlight_bond_lists.append([])

    try:
        img = Draw.MolsToGridImage(
            drawable_mols,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=legends,
            highlightAtomLists=highlight_atom_lists if highlight else None,
            highlightBondLists=highlight_bond_lists,
        )
    except RuntimeError:
        logger.warning(
            "molsPerRow is too small to draw a large number of molecules; try to increase it."
        )
        return None

    if output_file:
        img.save(output_file)

    return img
