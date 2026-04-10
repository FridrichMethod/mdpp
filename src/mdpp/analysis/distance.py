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
from mdpp.core.trajectory import select_atom_indices, trajectory_time_ps

type DistanceBackend = str


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


def _pairwise_distances_numba(
    xyz: NDArray[np.float32],
    pairs: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute non-periodic pairwise distances using a Numba-parallel kernel.

    Parallelises the frame loop using ``prange``, giving ~5x speedup over
    mdtraj's single-threaded C/SSE kernel on multi-core machines.

    The kernel outputs float64 because Numba maps Python ``float()``
    casts to double precision in its type system.  Changing the output
    dtype would require a separate JIT specialization.  Callers cast
    the result to the resolved dtype after the kernel returns.

    **Limitations compared to mdtraj:**

    - No periodic boundary condition (minimum image convention) support.
      Use ``backend="mdtraj"`` if the trajectory has unit-cell information
      and PBC distances are required.
    - Returns float64 (mdtraj returns float32).

    Args:
        xyz: Coordinates of shape ``(n_frames, n_atoms, 3)``, typically
            ``traj.xyz`` in nm.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` in the same unit as
        *xyz* (nm for mdtraj trajectories).

    Raises:
        ValueError: If any pair index is out of range.
    """
    n_atoms = xyz.shape[1]
    if pairs.size > 0 and (np.any(pairs < 0) or np.any(pairs >= n_atoms)):
        raise ValueError(
            f"atom_pairs must contain indices in [0, {n_atoms}), "
            f"got range [{int(pairs.min())}, {int(pairs.max())}]."
        )

    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _kernel(
        xyz: NDArray[np.float32], pairs: NDArray[np.int_]
    ) -> NDArray[np.float64]:  # pragma: no cover - JIT-compiled
        n_frames = xyz.shape[0]
        n_pairs = pairs.shape[0]
        out = np.empty((n_frames, n_pairs), dtype=np.float64)
        for f in prange(n_frames):
            for k in range(n_pairs):
                i = pairs[k, 0]
                j = pairs[k, 1]
                dx = float(xyz[f, i, 0]) - float(xyz[f, j, 0])
                dy = float(xyz[f, i, 1]) - float(xyz[f, j, 1])
                dz = float(xyz[f, i, 2]) - float(xyz[f, j, 2])
                out[f, k] = np.sqrt(dx * dx + dy * dy + dz * dz)
        return out

    return _kernel(xyz, pairs)


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


def _pairwise_distances_cupy(
    xyz: NDArray[np.float32],
    pairs: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute non-periodic pairwise distances on GPU using CuPy.

    Transfers coordinates to GPU, performs vectorised fancy-index
    differencing, and returns the result on the host.

    **Limitations:**

    - No periodic boundary condition support.
    - Requires ``cupy`` (install with ``pip install cupy-cuda12x``).
    - GPU memory must fit the intermediate ``(n_frames, n_pairs, 3)``
      diffs array plus the output.

    Args:
        xyz: Coordinates of shape ``(n_frames, n_atoms, 3)``.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` as a host numpy
        array (float64).

    Raises:
        ImportError: If CuPy is not installed.
        ValueError: If any pair index is out of range.
    """
    n_atoms = xyz.shape[1]
    if pairs.size > 0 and (np.any(pairs < 0) or np.any(pairs >= n_atoms)):
        raise ValueError(
            f"atom_pairs must contain indices in [0, {n_atoms}), "
            f"got range [{int(pairs.min())}, {int(pairs.max())}]."
        )

    try:
        import cupy as cp
    except ImportError:
        raise ImportError(
            "CuPy is required for backend='cupy'. Install it with: pip install cupy-cuda12x"
        ) from None

    xyz_gpu = cp.asarray(xyz)
    pairs_gpu = cp.asarray(pairs)
    diffs = xyz_gpu[:, pairs_gpu[:, 0], :] - xyz_gpu[:, pairs_gpu[:, 1], :]
    distances = cp.sqrt(cp.sum(diffs * diffs, axis=2))
    return cp.asnumpy(distances).astype(np.float64)


def _pairwise_distances_torch(
    xyz: NDArray[np.float32],
    pairs: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute non-periodic pairwise distances using PyTorch.

    Uses CUDA if available, otherwise falls back to CPU.  The
    vectorised fancy-index approach is identical to the CuPy kernel.

    **Limitations:**

    - No periodic boundary condition support.
    - Requires ``torch``.

    Args:
        xyz: Coordinates of shape ``(n_frames, n_atoms, 3)``.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` as a host numpy
        array (float64).

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If any pair index is out of range.
    """
    n_atoms = xyz.shape[1]
    if pairs.size > 0 and (np.any(pairs < 0) or np.any(pairs >= n_atoms)):
        raise ValueError(
            f"atom_pairs must contain indices in [0, {n_atoms}), "
            f"got range [{int(pairs.min())}, {int(pairs.max())}]."
        )

    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for backend='torch'. Install it with: pip install torch"
        ) from None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        xyz_t = torch.as_tensor(np.ascontiguousarray(xyz), device=device)
        pairs_t = torch.as_tensor(pairs.astype(np.int64), device=device)
        diffs = xyz_t[:, pairs_t[:, 0], :] - xyz_t[:, pairs_t[:, 1], :]
        distances = torch.sqrt(torch.sum(diffs * diffs, dim=2))
    return distances.cpu().numpy().astype(np.float64)


def _pairwise_distances_jax(
    xyz: NDArray[np.float32],
    pairs: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute non-periodic pairwise distances using JAX.

    JAX auto-selects the best available backend (GPU > TPU > CPU).
    The XLA compiler fuses the element-wise operations into a single
    kernel, giving performance comparable to the CuPy kernel on GPU.

    **Limitations:**

    - No periodic boundary condition support.
    - Requires ``jax``.

    Args:
        xyz: Coordinates of shape ``(n_frames, n_atoms, 3)``.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` as a host numpy
        array (float64).

    Raises:
        ImportError: If JAX is not installed.
        ValueError: If any pair index is out of range.
    """
    n_atoms = xyz.shape[1]
    if pairs.size > 0 and (np.any(pairs < 0) or np.any(pairs >= n_atoms)):
        raise ValueError(
            f"atom_pairs must contain indices in [0, {n_atoms}), "
            f"got range [{int(pairs.min())}, {int(pairs.max())}]."
        )

    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for backend='jax'. Install it with: pip install jax[cuda12]"
        ) from None

    xyz_j = jnp.asarray(xyz)
    pairs_j = jnp.asarray(pairs)
    diffs = xyz_j[:, pairs_j[:, 0], :] - xyz_j[:, pairs_j[:, 1], :]
    distances = jnp.sqrt(jnp.sum(diffs * diffs, axis=2))
    return np.asarray(distances).astype(np.float64)


_VALID_BACKENDS = ("mdtraj", "numba", "cupy", "torch", "jax")


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
    if backend == "numba":
        return np.asarray(_pairwise_distances_numba(traj.xyz, pairs), dtype=resolved)
    if backend == "cupy":
        return np.asarray(_pairwise_distances_cupy(traj.xyz, pairs), dtype=resolved)
    if backend == "torch":
        return np.asarray(_pairwise_distances_torch(traj.xyz, pairs), dtype=resolved)
    if backend == "jax":
        return np.asarray(_pairwise_distances_jax(traj.xyz, pairs), dtype=resolved)
    raise ValueError(f"Unknown backend {backend!r}. Use one of {_VALID_BACKENDS!r}.")


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
