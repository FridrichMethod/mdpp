"""Pairwise distance analysis for molecular dynamics trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdpp.core.trajectory import select_atom_indices, trajectory_time_ps


@dataclass(frozen=True, slots=True)
class DistanceResult:
    """Per-frame pairwise distances."""

    time_ps: NDArray[np.float64]
    distances_nm: NDArray[np.float64]
    atom_pairs: NDArray[np.int_]

    @property
    def time_ns(self) -> NDArray[np.float64]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0

    @property
    def distances_angstrom(self) -> NDArray[np.float64]:
        """Return distances in Angstrom."""
        return self.distances_nm * 10.0


def compute_distances(
    traj: md.Trajectory,
    *,
    atom_pairs: ArrayLike,
    periodic: bool = True,
    timestep_ps: float | None = None,
) -> DistanceResult:
    """Compute pairwise distances between atom pairs over time.

    Args:
        traj: Input trajectory.
        atom_pairs: Array of shape ``(n_pairs, 2)`` with atom index pairs.
        periodic: Whether to apply periodic boundary conditions.
        timestep_ps: Optional frame timestep override in ps.

    Returns:
        DistanceResult with per-frame distances for each pair.
    """
    pairs = np.asarray(atom_pairs, dtype=np.int_)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("atom_pairs must have shape (n_pairs, 2).")

    distances = md.compute_distances(traj, pairs, periodic=periodic)
    return DistanceResult(
        time_ps=trajectory_time_ps(traj, timestep_ps=timestep_ps),
        distances_nm=np.asarray(distances, dtype=np.float64),
        atom_pairs=pairs,
    )


def compute_minimum_distance(
    traj: md.Trajectory,
    *,
    group1: str,
    group2: str,
    periodic: bool = True,
    timestep_ps: float | None = None,
) -> DistanceResult:
    """Compute the minimum distance between two atom groups per frame.

    All pairwise distances between ``group1`` and ``group2`` atoms are
    computed, and the minimum per frame is returned.

    Args:
        traj: Input trajectory.
        group1: MDTraj selection string for the first group.
        group2: MDTraj selection string for the second group.
        periodic: Whether to apply periodic boundary conditions.
        timestep_ps: Optional frame timestep override in ps.

    Returns:
        DistanceResult where ``distances_nm`` has shape ``(n_frames, 1)``
        and ``atom_pairs`` contains the closest pair at frame 0.
    """
    indices_1 = select_atom_indices(traj.topology, group1)
    indices_2 = select_atom_indices(traj.topology, group2)

    pairs = np.array(
        [(i, j) for i in indices_1 for j in indices_2],
        dtype=np.int_,
    )
    all_distances = md.compute_distances(traj, pairs, periodic=periodic)

    min_indices = np.argmin(all_distances, axis=1)
    min_distances = all_distances[np.arange(traj.n_frames), min_indices]

    closest_pair = pairs[min_indices[0]].reshape(1, 2)

    return DistanceResult(
        time_ps=trajectory_time_ps(traj, timestep_ps=timestep_ps),
        distances_nm=np.asarray(min_distances.reshape(-1, 1), dtype=np.float64),
        atom_pairs=closest_pair,
    )
