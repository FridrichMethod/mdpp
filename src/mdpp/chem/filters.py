"""Molecular scaffold extraction and structural filters."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold

_PAINS_CATALOG: FilterCatalog.FilterCatalog | None = None


def _get_pains_catalog() -> FilterCatalog.FilterCatalog:
    """Return (and lazily build) a cached PAINS filter catalog."""
    global _PAINS_CATALOG  # noqa: PLW0603
    if _PAINS_CATALOG is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def get_framework(
    mol: Chem.rdchem.Mol | str,
    *,
    generic: bool = False,
) -> Chem.rdchem.Mol | str:
    """Get the Murcko scaffold of a molecule.

    Args:
        mol: An RDKit molecule or a SMILES string.
        generic: If True, return the generic (all-carbon, all-single-bond)
            scaffold.

    Returns:
        The scaffold in the same type as the input (SMILES string or Mol).
    """
    if isinstance(mol, str):
        if generic:
            return Chem.MolToSmiles(
                MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(mol))
                )
            )
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(mol)

    if generic:
        return MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
    return MurckoScaffold.GetScaffoldForMol(mol)


def is_pains(mol: Chem.rdchem.Mol) -> bool:
    """Check whether a molecule matches any PAINS filter.

    PAINS (Pan Assay Interference Compounds) are frequent hitters in
    high-throughput screens that act through non-specific mechanisms.

    Args:
        mol: An RDKit molecule.

    Returns:
        True if the molecule matches at least one PAINS pattern.
    """
    return _get_pains_catalog().HasMatch(mol)
