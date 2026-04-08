"""Trajectory loading and selection helpers based on MDTraj."""

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import Pool

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
    n_frames: int | None = None,
    atom_selection: str | None = None,
) -> md.Trajectory:
    """Load a single trajectory and optionally atom-slice it.

    When *n_frames* is provided, uses :func:`mdtraj.iterload` to stream
    chunks from disk and stops as soon as *n_frames* frames (after stride)
    have been collected, avoiding reading the full file into memory.

    Args:
        trajectory_path: Path to trajectory file (for example, ``.xtc``).
        topology_path: Optional topology path (for example, ``.pdb``).
        stride: Frame stride.
        n_frames: Optional maximum number of frames to load (after stride).
            If ``None``, the entire trajectory is loaded.
        atom_selection: Optional MDTraj selection for atom slicing after load.

    Returns:
        Loaded (and optionally sliced) trajectory.

    Raises:
        ValueError: If ``stride`` is less than 1 or ``n_frames`` is less than 1.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if n_frames is not None and n_frames < 1:
        raise ValueError("n_frames must be >= 1.")

    top = None if topology_path is None else str(topology_path)

    if n_frames is None:
        trajectory = md.load(str(trajectory_path), top=top, stride=stride)
    else:
        chunks: list[md.Trajectory] = []
        loaded = 0
        chunk_size = min(n_frames, 1000)
        for chunk in md.iterload(str(trajectory_path), top=top, stride=stride, chunk=chunk_size):
            need = n_frames - loaded
            if chunk.n_frames >= need:
                chunks.append(chunk[:need])
                loaded += need
                break
            chunks.append(chunk)
            loaded += chunk.n_frames
        trajectory = md.join(chunks)

    if atom_selection is None:
        return trajectory

    atom_indices = select_atom_indices(trajectory.topology, atom_selection)
    return trajectory.atom_slice(atom_indices)


def _load_trajectory_worker(
    args: tuple[str, str | None, int, int | None, str | None],
) -> md.Trajectory:
    """Worker function for parallel trajectory loading (must be picklable)."""
    traj_path, top_path, stride, n_frames, atom_selection = args
    return load_trajectory(
        trajectory_path=traj_path,
        topology_path=top_path,
        stride=stride,
        n_frames=n_frames,
        atom_selection=atom_selection,
    )


def load_trajectories(
    trajectory_paths: Sequence[PathLike],
    *,
    topology_paths: Sequence[PathLike | None] | None = None,
    stride: int = 1,
    n_frames: int | None = None,
    atom_selection: str | None = None,
    max_workers: int | None = None,
) -> list[md.Trajectory]:
    """Load a list of trajectories with a shared interface.

    When *max_workers* is set, trajectories are loaded in parallel using
    :class:`multiprocessing.Pool` (process-based parallelism).

    Why processes instead of threads:
        mdtraj's C-level XTC/TRR parsers hold the GIL during decoding, so
        threads cannot achieve true concurrency. In benchmarks on 6 replicas
        (stride=10, 1000 frames each), threads gave only ~1x speedup while
        processes achieved ~6x.

    Why ``multiprocessing.Pool`` instead of ``ProcessPoolExecutor``:
        Both perform identically in benchmarks for this workload. ``Pool`` is
        chosen for its simpler API (``map`` returns results directly) and
        ``maxtasksperchild`` support, which can guard against memory leaks
        from large trajectory allocations. The ``Future`` abstraction that
        ``concurrent.futures`` provides is unnecessary for a pure map
        operation.

    Args:
        trajectory_paths: Trajectory paths.
        topology_paths: Optional topology paths. If provided, must match
            ``trajectory_paths`` length.
        stride: Frame stride.
        n_frames: Optional maximum number of frames per trajectory (after stride).
        atom_selection: Optional atom selection for slicing.
        max_workers: If set, load trajectories in parallel using processes.
            The value controls the maximum number of concurrent worker
            processes. If ``None``, trajectories are loaded sequentially.

    Returns:
        Loaded trajectories in the same order as ``trajectory_paths``.

    Raises:
        ValueError: If ``topology_paths`` length does not match trajectories.
    """
    if topology_paths is None:
        topology_paths = [None] * len(trajectory_paths)
    elif len(topology_paths) != len(trajectory_paths):
        raise ValueError("trajectory_paths and topology_paths must have the same length.")

    args_list = [
        (
            str(traj_path),
            None if top_path is None else str(top_path),
            stride,
            n_frames,
            atom_selection,
        )
        for traj_path, top_path in zip(trajectory_paths, topology_paths, strict=True)
    ]

    if max_workers is None:
        return [_load_trajectory_worker(a) for a in args_list]

    with Pool(processes=max_workers) as pool:
        return pool.map(_load_trajectory_worker, args_list)


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
