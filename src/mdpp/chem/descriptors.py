"""Molecular descriptor calculation and filtering utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

BUILTIN_DESC_NAMES: set[str] = {name for name, _ in Descriptors.descList}

COMMON_DESC_NAMES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "NumHAcceptors",
    "NumHDonors",
    "FractionCSP3",
    "NumRotatableBonds",
    "RingCount",
    "TPSA",
    "qed",
)


def calc_descs(
    mol: Chem.rdchem.Mol,
    *,
    desc_names: Sequence[str] = COMMON_DESC_NAMES,
) -> float | tuple[float, ...]:
    """Calculate molecular descriptors.

    Default descriptors include Lipinski rule-of-five properties
    (``MolWt``, ``MolLogP``, ``NumHAcceptors``, ``NumHDonors``) and
    other common descriptors (``FractionCSP3``, ``NumRotatableBonds``,
    ``RingCount``, ``TPSA``, ``qed``).

    Args:
        mol: An RDKit molecule.
        desc_names: Descriptor names to calculate. Must be a subset of
            ``BUILTIN_DESC_NAMES``.

    Returns:
        A single float when one descriptor is requested, otherwise a tuple
        of floats in the same order as *desc_names*.

    Raises:
        KeyError: If any name in *desc_names* is not a valid RDKit descriptor.
    """
    if invalid := set(desc_names) - BUILTIN_DESC_NAMES:
        raise KeyError(f"Invalid descriptor names: {invalid}")

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    result = calc.CalcDescriptors(mol)
    return result[0] if len(desc_names) == 1 else result


def filt_descs(
    mol: Chem.rdchem.Mol,
    *,
    filt: dict[str, tuple[float, float]],
) -> bool:
    """Filter a molecule based on descriptor value ranges.

    Args:
        mol: An RDKit molecule.
        filt: Mapping of descriptor names to ``(lower, upper)`` bounds.
            An empty dict lets every molecule pass.

    Returns:
        True if all descriptors fall within their specified ranges.
    """
    if not filt:
        return True

    descs = np.array(calc_descs(mol, desc_names=tuple(filt.keys())))
    bounds = np.array(list(filt.values()))

    return bool(np.all((bounds[:, 0] <= descs) & (descs <= bounds[:, 1])))
