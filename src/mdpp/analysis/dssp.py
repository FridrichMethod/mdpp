"""Secondary structure assignment via DSSP."""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DSSPResult:
    """Per-frame secondary structure assignments.

    Attributes:
        assignments: Character array of shape ``(n_frames, n_residues)``
            with DSSP codes (``"H"``, ``"E"``, ``"C"`` when simplified, or
            full 8-state codes otherwise).
        residue_ids: Residue sequence IDs corresponding to columns.
        frequency: Array of shape ``(n_residues, n_categories)`` giving the
            fraction of frames each residue spends in each secondary structure
            category.
        categories: List of unique category labels matching the last axis
            of ``frequency``.
    """

    assignments: NDArray[np.str_]
    residue_ids: NDArray[np.int_]
    frequency: NDArray[np.float64]
    categories: list[str]


def compute_dssp(
    traj: md.Trajectory,
    *,
    simplified: bool = True,
) -> DSSPResult:
    """Compute per-residue secondary structure assignments across frames.

    Args:
        traj: Input trajectory.
        simplified: If ``True``, use 3-state classification (H=helix,
            E=sheet, C=coil). Otherwise use the full 8-state DSSP codes.

    Returns:
        DSSPResult with per-frame assignments and per-residue frequencies.
    """
    assignments = md.compute_dssp(traj, simplified=simplified)
    assignments = np.asarray(assignments, dtype=np.str_)

    residue_ids = np.array(
        [residue.resSeq for residue in traj.topology.residues],
        dtype=np.int_,
    )

    if simplified:
        categories = ["H", "E", "C"]
    else:
        categories = sorted({str(code) for code in assignments.ravel()})

    n_residues = assignments.shape[1]
    frequency = np.zeros((n_residues, len(categories)), dtype=np.float64)
    for cat_index, cat in enumerate(categories):
        frequency[:, cat_index] = np.mean(assignments == cat, axis=0)

    return DSSPResult(
        assignments=assignments,
        residue_ids=residue_ids,
        frequency=frequency,
        categories=categories,
    )
