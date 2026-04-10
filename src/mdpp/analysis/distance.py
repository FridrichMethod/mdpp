"""Pairwise distance analysis for molecular dynamics trajectories.

Five backends are available for pairwise distance computation:

+----------+------------------+-----+----------------------------+
| Backend  | Device           | PBC | Dependency                 |
+==========+==================+=====+============================+
| mdtraj   | CPU (1 thread)   | Yes | built-in                   |
| numba    | CPU (all cores)  | No  | built-in (numba)           |
| cupy     | GPU (CUDA)       | No  | ``pip install cupy-cuda12x``|
| torch    | GPU (CUDA) / CPU | No  | ``pip install torch``      |
| jax      | GPU / TPU / CPU  | No  | ``pip install jax[cuda12]``|
+----------+------------------+-----+----------------------------+

The GPU backends (cupy, torch, jax) use vectorised fancy-index
differencing.  This materialises an intermediate array of shape
``(n_frames, n_pairs, 3)`` on the device, so GPU memory must be
sufficient.  The Numba backend computes element-by-element with no
intermediate allocation, making it the fastest at small-to-medium
scales and competitive even at large scales.

Benchmark results (24-core CPU, NVIDIA GPU)::

    1K frames x 100 atoms (4,950 pairs)
      numba  0.005s  2.6x    mdtraj 0.013s  1.0x

    3K frames x 200 atoms (19,900 pairs)
      numba  0.021s  7.9x    mdtraj 0.166s  1.0x

    3K frames x 400 atoms (79,800 pairs)
      numba  0.059s 10.4x    mdtraj 0.617s  1.0x

GPU backends approach Numba at higher pair counts where device
parallelism offsets transfer overhead.  Use ``backend="numba"``
as the default for non-periodic featurisation workloads.
"""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdpp._dtype import resolve_dtype
from mdpp._types import DtypeArg
from mdpp.analysis._backends import DistanceBackend
from mdpp.analysis._backends._distances import distance_backends
from mdpp.core.trajectory import select_atom_indices, trajectory_time_ps

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DistanceResult:
    """Per-frame pairwise distances."""

    time_ps: NDArray[np.floating]
    distances_nm: NDArray[np.floating]
    atom_pairs: NDArray[np.int_]

    @property
    def time_ns(self) -> NDArray[np.floating]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0

    @property
    def distances_angstrom(self) -> NDArray[np.floating]:
        """Return distances in Angstrom."""
        return self.distances_nm * 10.0


# ---------------------------------------------------------------------------
# mdtraj backend (unique signature: supports PBC)
# ---------------------------------------------------------------------------


def _pairwise_distances_mdtraj(
    traj: md.Trajectory,
    pairs: NDArray[np.int_],
    *,
    periodic: bool,
    dtype: DtypeArg = None,
) -> NDArray[np.floating]:
    """Compute pairwise distances using mdtraj's optimised C/SSE kernel.

    Supports periodic boundary conditions via minimum image convention
    when the trajectory contains unit-cell information.

    Args:
        traj: Atom-sliced trajectory.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.
        periodic: Whether to apply minimum image convention.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        Distances of shape ``(n_frames, n_pairs)``.
    """
    resolved = resolve_dtype(dtype)
    return np.asarray(
        md.compute_distances(traj, pairs, periodic=periodic),
        dtype=resolved,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _compute_pairwise_distances(
    traj: md.Trajectory,
    pairs: NDArray[np.int_],
    *,
    backend: DistanceBackend = "mdtraj",
    periodic: bool = True,
    dtype: DtypeArg = None,
) -> NDArray[np.floating]:
    """Dispatch pairwise distance computation to the selected backend.

    Five backends are available:

    ``"mdtraj"`` (default)
        mdtraj's optimised C/SSE ``compute_distances``.  Supports
        periodic boundary conditions via minimum image convention when
        the trajectory contains unit-cell information.  Single-threaded.

    ``"numba"``
        Numba-parallel kernel that distributes frames across all CPU
        cores.  ~5x faster than mdtraj for non-periodic systems on
        multi-core machines.  **Does not support periodic boundary
        conditions** -- the *periodic* parameter is ignored.

    ``"cupy"``
        GPU-accelerated kernel using CuPy vectorised operations.
        Requires ``cupy`` (``pip install cupy-cuda12x``).
        **Does not support periodic boundary conditions.**

    ``"torch"``
        GPU-accelerated kernel using PyTorch.  Uses CUDA if available,
        otherwise CPU.  Requires ``torch``.
        **Does not support periodic boundary conditions.**

    ``"jax"``
        GPU-accelerated kernel using JAX/XLA.  Auto-selects the best
        available device.  Requires ``jax``.
        **Does not support periodic boundary conditions.**

    Args:
        traj: Input trajectory.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.
        backend: Distance computation backend.
        periodic: Whether to apply minimum image convention. Only
            effective with ``backend="mdtraj"``.
        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        Distances of shape ``(n_frames, n_pairs)``.

    Raises:
        ValueError: If an unknown backend is requested.
    """
    resolved = resolve_dtype(dtype)
    if backend == "mdtraj":
        return _pairwise_distances_mdtraj(traj, pairs, periodic=periodic, dtype=resolved)
    all_backends = ("mdtraj", *distance_backends.names)
    try:
        compute_fn = distance_backends.get(backend)
    except ValueError:
        raise ValueError(f"Unknown backend {backend!r}. Use one of {all_backends!r}.") from None
    return np.asarray(compute_fn(traj.xyz, pairs), dtype=resolved)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_distances(
    traj: md.Trajectory,
    *,
    atom_pairs: ArrayLike,
    periodic: bool = True,
    backend: DistanceBackend = "mdtraj",
    timestep_ps: float | None = None,
    dtype: DtypeArg = None,
) -> DistanceResult:
    """Compute pairwise distances between atom pairs over time.

    Args:
        traj: Input trajectory.
        atom_pairs: Array of shape ``(n_pairs, 2)`` with atom index pairs.
        periodic: Whether to apply periodic boundary conditions.
        backend: Distance computation backend. ``"mdtraj"`` (default,
            PBC-capable), ``"numba"`` (CPU-parallel), ``"cupy"``/
            ``"torch"``/``"jax"`` (GPU-accelerated). Non-mdtraj
            backends do not support periodic boundary conditions.
        timestep_ps: Optional frame timestep override in ps.
        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        DistanceResult with per-frame distances for each pair.
    """
    resolved = resolve_dtype(dtype)
    pairs = np.asarray(atom_pairs, dtype=np.int_)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("atom_pairs must have shape (n_pairs, 2).")

    distances = _compute_pairwise_distances(
        traj,
        pairs,
        backend=backend,
        periodic=periodic,
        dtype=resolved,
    )
    return DistanceResult(
        time_ps=trajectory_time_ps(traj, timestep_ps=timestep_ps, dtype=resolved),
        distances_nm=distances,
        atom_pairs=pairs,
    )


def compute_minimum_distance(
    traj: md.Trajectory,
    *,
    group1: str,
    group2: str,
    periodic: bool = True,
    backend: DistanceBackend = "mdtraj",
    timestep_ps: float | None = None,
    dtype: DtypeArg = None,
) -> DistanceResult:
    """Compute the minimum distance between two atom groups per frame.

    All pairwise distances between ``group1`` and ``group2`` atoms are
    computed, and the minimum per frame is returned.

    Args:
        traj: Input trajectory.
        group1: MDTraj selection string for the first group.
        group2: MDTraj selection string for the second group.
        periodic: Whether to apply periodic boundary conditions.
        backend: Distance computation backend. ``"mdtraj"`` (default,
            PBC-capable), ``"numba"`` (CPU-parallel), ``"cupy"``/
            ``"torch"``/``"jax"`` (GPU-accelerated). Non-mdtraj
            backends do not support periodic boundary conditions.
        timestep_ps: Optional frame timestep override in ps.
        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        DistanceResult where ``distances_nm`` has shape ``(n_frames, 1)``
        and ``atom_pairs`` contains the closest pair at frame 0.
    """
    resolved = resolve_dtype(dtype)
    indices_1 = select_atom_indices(traj.topology, group1)
    indices_2 = select_atom_indices(traj.topology, group2)

    pairs = np.array(
        [(i, j) for i in indices_1 for j in indices_2],
        dtype=np.int_,
    )
    all_distances = _compute_pairwise_distances(
        traj,
        pairs,
        backend=backend,
        periodic=periodic,
        dtype=resolved,
    )

    min_indices = np.argmin(all_distances, axis=1)
    min_distances = all_distances[np.arange(traj.n_frames), min_indices]

    closest_pair = pairs[min_indices[0]].reshape(1, 2)

    return DistanceResult(
        time_ps=trajectory_time_ps(traj, timestep_ps=timestep_ps, dtype=resolved),
        distances_nm=min_distances.reshape(-1, 1),
        atom_pairs=closest_pair,
    )
