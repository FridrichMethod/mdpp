"""Trajectory loading and selection helpers based on MDTraj."""

from __future__ import annotations

from collections.abc import Sequence

import mdtraj as md
import numpy as np
from numpy.typing import NDArray

from mdpp._types import PathLike


def select_atom_indices(topology: md.Topology, selection: str) -> NDArray[np.int_]:
    """Return atom indices selected by an MDTraj DSL selection.

    Args:
        topology: Trajectory topology.
        selection: MDTraj selection string (for example, ``"name CA"``).

    Returns:
        Atom indices matching the selection.

    Raises:
        ValueError: If the selection matches no atoms.
    """
    atom_indices = topology.select(selection).astype(np.int_)
    if atom_indices.size == 0:
        raise ValueError(f"Selection {selection!r} matched no atoms.")
    return atom_indices


def residue_ids_from_indices(
    topology: md.Topology,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Map atom indices to residue sequence IDs.

    Args:
        topology: Trajectory topology.
        atom_indices: Atom indices to map.

    Returns:
        Residue IDs for each atom index.
    """
    return np.array(
        [topology.atom(int(atom_index)).residue.resSeq for atom_index in atom_indices],
        dtype=np.int_,
    )


def trajectory_time_ps(
    traj: md.Trajectory,
    *,
    timestep_ps: float | None = None,
) -> NDArray[np.float64]:
    """Return per-frame time values in picoseconds.

    Args:
        traj: Input trajectory.
        timestep_ps: Optional fixed timestep to enforce. If provided, generated
            time values are ``np.arange(n_frames) * timestep_ps``.

    Returns:
        Time array in picoseconds.
    """
    if timestep_ps is not None:
        return np.arange(traj.n_frames, dtype=np.float64) * float(timestep_ps)

    time_ps = np.asarray(traj.time, dtype=np.float64)
    if time_ps.shape[0] == traj.n_frames:
        return time_ps
    return np.arange(traj.n_frames, dtype=np.float64)


def load_trajectory(
    trajectory_path: PathLike,
    *,
    topology_path: PathLike | None = None,
    stride: int = 1,
    atom_selection: str | None = None,
) -> md.Trajectory:
    """Load a single trajectory and optionally atom-slice it.

    Args:
        trajectory_path: Path to trajectory file (for example, ``.xtc``).
        topology_path: Optional topology path (for example, ``.pdb``).
        stride: Frame stride.
        atom_selection: Optional MDTraj selection for atom slicing after load.

    Returns:
        Loaded (and optionally sliced) trajectory.

    Raises:
        ValueError: If ``stride`` is less than 1.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1.")

    trajectory = md.load(
        str(trajectory_path),
        top=None if topology_path is None else str(topology_path),
        stride=stride,
    )
    if atom_selection is None:
        return trajectory

    atom_indices = select_atom_indices(trajectory.topology, atom_selection)
    return trajectory.atom_slice(atom_indices)


def load_trajectories(
    trajectory_paths: Sequence[PathLike],
    *,
    topology_paths: Sequence[PathLike | None] | None = None,
    stride: int = 1,
    atom_selection: str | None = None,
) -> list[md.Trajectory]:
    """Load a list of trajectories with a shared interface.

    Args:
        trajectory_paths: Trajectory paths.
        topology_paths: Optional topology paths. If provided, must match
            ``trajectory_paths`` length.
        stride: Frame stride.
        atom_selection: Optional atom selection for slicing.

    Returns:
        Loaded trajectories.

    Raises:
        ValueError: If ``topology_paths`` length does not match trajectories.
    """
    if topology_paths is None:
        topology_paths = [None] * len(trajectory_paths)
    elif len(topology_paths) != len(trajectory_paths):
        raise ValueError("trajectory_paths and topology_paths must have the same length.")

    return [
        load_trajectory(
            trajectory_path=trajectory_path,
            topology_path=topology_path,
            stride=stride,
            atom_selection=atom_selection,
        )
        for trajectory_path, topology_path in zip(trajectory_paths, topology_paths, strict=True)
    ]


def align_trajectory(
    traj: md.Trajectory,
    *,
    atom_selection: str = "name CA",
    reference_frame: int = 0,
    inplace: bool = False,
) -> md.Trajectory:
    """Align a trajectory to a reference frame.

    Args:
        traj: Input trajectory.
        atom_selection: Atoms used for alignment.
        reference_frame: Reference frame index.
        inplace: Whether to align ``traj`` in place.

    Returns:
        The aligned trajectory.

    Raises:
        ValueError: If ``reference_frame`` is out of range.
    """
    if not 0 <= reference_frame < traj.n_frames:
        raise ValueError(
            f"reference_frame must be in [0, {traj.n_frames - 1}], got {reference_frame}."
        )

    aligned = traj if inplace else traj[:]
    atom_indices = select_atom_indices(aligned.topology, atom_selection)
    aligned.superpose(aligned, frame=reference_frame, atom_indices=atom_indices)
    return aligned
