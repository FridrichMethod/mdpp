"""Conformational clustering from RMSD matrices."""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numpy.typing import NDArray

from mdpp._dtype import resolve_dtype
from mdpp._types import DtypeArg
from mdpp.core.trajectory import select_atom_indices


@dataclass(frozen=True, slots=True)
class RMSDMatrixResult:
    """Pairwise RMSD matrix between trajectory frames."""

    rmsd_matrix_nm: NDArray[np.floating]
    atom_indices: NDArray[np.int_]

    @property
    def rmsd_matrix_angstrom(self) -> NDArray[np.floating]:
        """Return the RMSD matrix in Angstrom."""
        return self.rmsd_matrix_nm * 10.0


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """Conformational clustering output."""

    labels: NDArray[np.int_]
    n_clusters: int
    medoid_frames: NDArray[np.int_]


def compute_rmsd_matrix(
    traj: md.Trajectory,
    *,
    atom_selection: str = "backbone",
    dtype: DtypeArg = None,
) -> RMSDMatrixResult:
    """Compute an all-vs-all RMSD matrix between trajectory frames.

    Args:
        traj: Input trajectory.
        atom_selection: Atoms used for RMSD calculation.
        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        RMSDMatrixResult with a symmetric ``(n_frames, n_frames)`` matrix.
    """
    resolved = resolve_dtype(dtype)
    atom_indices = select_atom_indices(traj.topology, atom_selection)
    n_frames = traj.n_frames
    rmsd_matrix = np.zeros((n_frames, n_frames), dtype=resolved)

    for i in range(n_frames):
        rmsd_matrix[i] = md.rmsd(traj, traj, frame=i, atom_indices=atom_indices)

    return RMSDMatrixResult(
        rmsd_matrix_nm=rmsd_matrix,
        atom_indices=atom_indices,
    )


def cluster_conformations(
    rmsd_matrix: NDArray[np.floating],
    *,
    method: str = "gromos",
    cutoff_nm: float = 0.15,
) -> ClusteringResult:
    """Cluster trajectory frames from a pairwise RMSD matrix.

    Args:
        rmsd_matrix: Symmetric pairwise RMSD matrix of shape ``(n, n)``
            in nm.
        method: Clustering method. ``"gromos"`` uses the GROMOS algorithm
            (largest-cluster-first greedy assignment).
        cutoff_nm: RMSD cutoff in nm for the GROMOS algorithm.

    Returns:
        ClusteringResult with per-frame labels and medoid frame indices.

    Raises:
        ValueError: If an unsupported method is specified.
    """
    if method != "gromos":
        raise ValueError(f"Unsupported clustering method: {method!r}. Use 'gromos'.")

    n_frames = rmsd_matrix.shape[0]
    labels = np.full(n_frames, -1, dtype=np.int_)
    assigned = np.zeros(n_frames, dtype=bool)
    cluster_id = 0
    medoids: list[int] = []

    while not np.all(assigned):
        neighbor_counts = np.zeros(n_frames, dtype=np.int_)
        for i in range(n_frames):
            if assigned[i]:
                continue
            neighbor_counts[i] = np.sum((rmsd_matrix[i] <= cutoff_nm) & (~assigned))

        center = int(np.argmax(neighbor_counts))
        members = (~assigned) & (rmsd_matrix[center] <= cutoff_nm)
        labels[members] = cluster_id
        assigned[members] = True
        medoids.append(center)
        cluster_id += 1

    return ClusteringResult(
        labels=labels,
        n_clusters=cluster_id,
        medoid_frames=np.array(medoids, dtype=np.int_),
    )
