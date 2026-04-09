"""Trajectory loading and selection helpers based on MDTraj."""

from __future__ import annotations

from collections.abc import Sequence

import mdtraj as md
import numpy as np
from numpy.typing import NDArray

from mdpp._dtype import resolve_dtype
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
    dtype: type[np.floating] | np.dtype[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Return per-frame time values in picoseconds.

    Args:
        traj: Input trajectory.
        timestep_ps: Optional fixed timestep to enforce. If provided, generated
            time values are ``np.arange(n_frames) * timestep_ps``.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        Time array in picoseconds.
    """
    resolved = resolve_dtype(dtype)
    if timestep_ps is not None:
        return np.arange(traj.n_frames, dtype=resolved) * float(timestep_ps)

    time_ps = np.asarray(traj.time, dtype=resolved)
    if time_ps.shape[0] == traj.n_frames:
        return time_ps
    return np.arange(traj.n_frames, dtype=resolved)


def load_trajectory(
    trajectory_path: PathLike,
    *,
    topology_path: PathLike | None = None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    atom_selection: str | None = None,
) -> md.Trajectory:
    """Load a single trajectory with optional frame and atom selection.

    Frame selection follows Python's ``range(start, stop, stride)``
    convention: *start* is included, *stop* is excluded, and *stride*
    controls the step size.  All three refer to **raw frame indices** in
    the trajectory file.

    When *atom_selection* is provided, the selected atom indices are
    passed directly to the underlying mdtraj reader so that only those
    atoms are read from disk.  This avoids loading the full-atom
    trajectory into memory.

    Args:
        trajectory_path: Path to trajectory file (for example, ``.xtc``).
        topology_path: Optional topology path (for example, ``.pdb``).
        start: First raw frame index to load (inclusive). Default is 0.
        stop: Raw frame index at which to stop loading (exclusive). If
            ``None``, read to the end of the file.
        stride: Frame stride (step size). Default is 1.
        atom_selection: Optional MDTraj atom selection string. Matching
            atoms are loaded directly from disk (no post-load slicing).

    Returns:
        Loaded trajectory containing only the selected frames and atoms.

    Raises:
        ValueError: If ``stride`` is less than 1, ``start`` is negative,
            or ``stop`` is not greater than ``start``.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if start < 0:
        raise ValueError("start must be >= 0.")
    if stop is not None and stop <= start:
        raise ValueError("stop must be greater than start.")

    top = None if topology_path is None else str(topology_path)
    topology = md.load_topology(top) if top is not None else md.load_topology(str(trajectory_path))
    atom_indices = (
        select_atom_indices(topology, atom_selection) if atom_selection is not None else None
    )

    if start == 0 and stop is None:
        return md.load(str(trajectory_path), top=top, stride=stride, atom_indices=atom_indices)

    n_frames = len(range(start, stop, stride)) if stop is not None else None
    with md.open(str(trajectory_path)) as fh:
        if start > 0:
            try:
                fh.seek(start)
            except OSError:
                n_atoms = len(atom_indices) if atom_indices is not None else topology.n_atoms
                sliced_top = topology.subset(atom_indices) if atom_indices is not None else topology
                return md.Trajectory(
                    xyz=np.empty((0, n_atoms, 3), dtype=np.float32),
                    topology=sliced_top,
                )
        return fh.read_as_traj(
            topology, n_frames=n_frames, stride=stride, atom_indices=atom_indices
        )


def _load_trajectory_worker(
    args: tuple[str, str | None, int, int | None, int, str | None],
) -> md.Trajectory:
    """Worker function for parallel trajectory loading (must be picklable)."""
    traj_path, top_path, start, stop, stride, atom_selection = args
    return load_trajectory(
        trajectory_path=traj_path,
        topology_path=top_path,
        start=start,
        stop=stop,
        stride=stride,
        atom_selection=atom_selection,
    )


def load_trajectories(
    trajectory_paths: Sequence[PathLike],
    *,
    topology_paths: Sequence[PathLike | None] | None = None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    atom_selection: str | None = None,
    max_workers: int | None = None,
) -> list[md.Trajectory]:
    """Load a list of trajectories with a shared interface.

    Frame selection follows Python's ``range(start, stop, stride)``
    convention: *start* is included, *stop* is excluded, and *stride*
    controls the step size.  All three refer to **raw frame indices**.

    When *max_workers* is set, trajectories are loaded in parallel using
    :class:`multiprocessing.Pool` (process-based parallelism).

    Why processes instead of threads:
        mdtraj's C-level XTC/TRR parsers hold the GIL during frame
        decoding, so threads cannot run concurrently on the CPU-bound
        parsing step. Benchmarks on 6 replicas (stride=10, 1000 frames
        each, ~5000 atoms) show:

        ============  ======  =========  ===========
        Method        Time    Speedup    RSS delta
        ============  ======  =========  ===========
        Sequential    9.7 s   1.0x       +16.8 MB
        Threads (6)   4.5 s   2.2x       +7.7 MB
        mp.Pool (6)   0.9 s   11.2x      +0.0 MB
        ============  ======  =========  ===========

        Processes win on both speed and memory. Worker processes allocate
        trajectory data in their own address space; when the pool closes
        that memory is fully released to the OS, leaving zero RSS growth
        in the parent. Threads allocate within the parent and rely on
        Python's allocator to (possibly) return pages.

    Why ``multiprocessing.Pool`` instead of ``ProcessPoolExecutor``:
        Both perform identically in benchmarks for this workload. ``Pool``
        is chosen for its simpler API (``map`` returns results directly)
        and ``maxtasksperchild`` support, which can guard against memory
        leaks from large trajectory allocations.

    Args:
        trajectory_paths: Trajectory paths.
        topology_paths: Optional topology paths. If provided, must match
            ``trajectory_paths`` length.
        start: First raw frame index to load (inclusive). Default is 0.
        stop: Raw frame index at which to stop (exclusive). If ``None``,
            read to the end of each file.
        stride: Frame stride (step size). Default is 1.
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
            start,
            stop,
            stride,
            atom_selection,
        )
        for traj_path, top_path in zip(trajectory_paths, topology_paths, strict=True)
    ]

    if max_workers is None:
        return [_load_trajectory_worker(a) for a in args_list]

    ctx = __import__("multiprocessing").get_context("spawn")
    with ctx.Pool(processes=max_workers) as pool:
        return pool.map(_load_trajectory_worker, args_list)


def align_trajectory(
    traj: md.Trajectory,
    *,
    atom_selection: str = "name CA",
    reference_frame: int = 0,
    inplace: bool = False,
) -> md.Trajectory:
    """Align a trajectory to a reference frame.

    ``md.Trajectory.superpose`` modifies coordinates in place.  When
    ``inplace=False`` (the default), only the ``xyz`` array is copied;
    topology and time are shared with the original trajectory.  This
    avoids the expensive ``deepcopy(topology)`` that ``traj[:]`` performs.

    Args:
        traj: Input trajectory.
        atom_selection: Atoms used for alignment.
        reference_frame: Reference frame index.
        inplace: If ``True``, align ``traj`` in place and return it.
            If ``False`` (default), return a new trajectory that shares
            topology and time but has its own aligned coordinates.

    Returns:
        The aligned trajectory.

    Raises:
        ValueError: If ``reference_frame`` is out of range.
    """
    if not 0 <= reference_frame < traj.n_frames:
        raise ValueError(
            f"reference_frame must be in [0, {traj.n_frames - 1}], got {reference_frame}."
        )

    if inplace:
        aligned = traj
    else:
        aligned = traj.slice(slice(None), copy=False)
        aligned.xyz = traj.xyz.copy()

    atom_indices = select_atom_indices(aligned.topology, atom_selection)
    aligned.superpose(aligned, frame=reference_frame, atom_indices=atom_indices)
    return aligned
