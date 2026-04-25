"""Pairwise distance computation backends.

Provides five backends for pairwise distances between atom pairs:

- ``mdtraj`` -- mdtraj's optimised C/SSE kernel (supports PBC).
- ``numba`` -- Numba-parallel CPU kernel.
- ``cupy`` -- CuPy vectorised operations on GPU.
- ``torch`` -- PyTorch vectorised operations on GPU/CPU.
- ``jax`` -- JAX/XLA vectorised operations on GPU/CPU.

All backends share the same positional signature ``(traj, pairs)`` and
return a floating-point numpy array of shape ``(n_frames, n_pairs)``
in the backend's **native** dtype (float32 for mdtraj / GPU backends,
float64 for numba).  The public :func:`compute_distances` wrapper
casts with ``copy=False`` so no redundant copy is made when the
resolved user dtype already matches.

The mdtraj backend additionally accepts a keyword-only ``periodic``
argument for minimum image convention; non-mdtraj backends do not
support periodic boundary conditions.
"""

from __future__ import annotations

from typing import Protocol

import mdtraj as md
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from mdpp.analysis._backends._imports import (
    clean_cupy_cache,
    clean_torch_cache,
    require_cupy,
    require_jax,
    require_torch,
)
from mdpp.analysis._backends._registry import BackendRegistry


class DistanceBackendFn(Protocol):
    """Callable signature for a pairwise distance backend.

    All registered backends accept a trajectory and an ``(n_pairs, 2)``
    array of 0-based atom-index pairs, returning an
    ``(n_frames, n_pairs)`` floating-point array in the backend's
    native dtype (typically float32).  The public
    :func:`compute_distances` wrapper casts with ``copy=False`` to
    the user-resolved dtype, so same-dtype returns do not duplicate
    memory.

    The mdtraj backend additionally accepts ``periodic`` as a
    keyword-only argument; all other backends silently ignore PBC.
    """

    def __call__(
        self,
        traj: md.Trajectory,
        pairs: NDArray[np.int_],
        *,
        periodic: bool = ...,
    ) -> NDArray[np.floating]: ...


def _validate_pairs(n_atoms: int, pairs: NDArray[np.int_]) -> None:
    """Raise ValueError if any pair index is out of range."""
    if pairs.size > 0 and (np.any(pairs < 0) or np.any(pairs >= n_atoms)):
        raise ValueError(
            f"atom_pairs must contain indices in [0, {n_atoms}), "
            f"got range [{int(pairs.min())}, {int(pairs.max())}]."
        )


def distances_mdtraj(
    traj: md.Trajectory,
    pairs: NDArray[np.int_],
    *,
    periodic: bool = False,
) -> NDArray[np.floating]:
    """Compute pairwise distances using mdtraj's optimised C/SSE kernel.

    Supports periodic boundary conditions via minimum image convention
    when the trajectory contains unit-cell information.

    Args:
        traj: Input trajectory.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.
        periodic: Whether to apply minimum image convention.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` in mdtraj's native
        float32.  The public :func:`compute_distances` wrapper casts
        with ``copy=False`` to the user-resolved dtype.
    """
    return md.compute_distances(traj, pairs, periodic=periodic)


def distances_numba(
    traj: md.Trajectory,
    pairs: NDArray[np.int_],
    *,
    periodic: bool = False,  # noqa: ARG001 - accepted for Protocol uniformity, ignored
) -> NDArray[np.floating]:
    """Compute non-periodic pairwise distances using a Numba-parallel kernel.

    Parallelises the frame loop using ``prange``, giving ~5x speedup over
    mdtraj's single-threaded C/SSE kernel on multi-core machines.

    Args:
        traj: Input trajectory.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.
        periodic: Ignored. Accepted for uniformity with
            :func:`distances_mdtraj`; this kernel does not support
            periodic boundary conditions.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` in **float32**.
        Intermediate math still promotes to C ``double`` via
        ``float()`` so precision matches mdtraj's float32 output;
        only the final store truncates to float32.  Half the
        memory of the old float64 output (critical at large
        ``n_frames * n_pairs``).

    Raises:
        ValueError: If any pair index is out of range.
    """
    _validate_pairs(traj.n_atoms, pairs)

    @njit(parallel=True, cache=True)
    def _kernel(
        xyz: NDArray[np.float32], pairs: NDArray[np.int_]
    ) -> NDArray[np.floating]:  # pragma: no cover - JIT-compiled
        n_frames = xyz.shape[0]
        n_pairs = pairs.shape[0]
        out = np.empty((n_frames, n_pairs), dtype=np.float32)
        for f in prange(n_frames):
            for k in range(n_pairs):
                i = pairs[k, 0]
                j = pairs[k, 1]
                dx = float(xyz[f, i, 0]) - float(xyz[f, j, 0])
                dy = float(xyz[f, i, 1]) - float(xyz[f, j, 1])
                dz = float(xyz[f, i, 2]) - float(xyz[f, j, 2])
                out[f, k] = np.sqrt(dx * dx + dy * dy + dz * dz)
        return out

    return _kernel(traj.xyz, pairs)


@clean_cupy_cache
def distances_cupy(
    traj: md.Trajectory,
    pairs: NDArray[np.int_],
    *,
    periodic: bool = False,  # noqa: ARG001 - accepted for Protocol uniformity, ignored
) -> NDArray[np.floating]:
    """Compute non-periodic pairwise distances on GPU using CuPy.

    Args:
        traj: Input trajectory.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.
        periodic: Ignored. Accepted for uniformity with
            :func:`distances_mdtraj`; this kernel does not support
            periodic boundary conditions.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` in float32 (cupy
        inherits the float32 of ``traj.xyz``).  The wrapper casts
        ``copy=False`` to the user-resolved dtype.

    Raises:
        ImportError: If CuPy is not installed.
        ValueError: If any pair index is out of range.
    """
    _validate_pairs(traj.n_atoms, pairs)
    cp = require_cupy()

    xyz_gpu = cp.asarray(traj.xyz)
    pairs_gpu = cp.asarray(pairs)
    diffs = xyz_gpu[:, pairs_gpu[:, 0], :] - xyz_gpu[:, pairs_gpu[:, 1], :]
    distances = cp.sqrt(cp.sum(diffs * diffs, axis=2))
    return cp.asnumpy(distances)


@clean_torch_cache
def distances_torch(
    traj: md.Trajectory,
    pairs: NDArray[np.int_],
    *,
    periodic: bool = False,  # noqa: ARG001 - accepted for Protocol uniformity, ignored
) -> NDArray[np.floating]:
    """Compute non-periodic pairwise distances using PyTorch.

    Uses CUDA if available, otherwise falls back to CPU.

    Args:
        traj: Input trajectory.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.
        periodic: Ignored. Accepted for uniformity with
            :func:`distances_mdtraj`; this kernel does not support
            periodic boundary conditions.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` in float32 (torch
        inherits the float32 of ``traj.xyz``).  The wrapper casts
        ``copy=False`` to the user-resolved dtype.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If any pair index is out of range.
    """
    _validate_pairs(traj.n_atoms, pairs)
    torch = require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ``inference_mode`` is strictly stronger than ``no_grad``: it
    # also disables view tracking and version counter increments,
    # which gives a small but real speedup on the fancy-indexing
    # path used here.  Both rule out grad bookkeeping.
    with torch.inference_mode():
        xyz_t = torch.as_tensor(np.ascontiguousarray(traj.xyz), device=device)
        pairs_t = torch.as_tensor(pairs.astype(np.int64), device=device)
        diffs = xyz_t[:, pairs_t[:, 0], :] - xyz_t[:, pairs_t[:, 1], :]
        distances = torch.sqrt(torch.sum(diffs * diffs, dim=2))
    return distances.cpu().numpy()


def distances_jax(
    traj: md.Trajectory,
    pairs: NDArray[np.int_],
    *,
    periodic: bool = False,  # noqa: ARG001 - accepted for Protocol uniformity, ignored
) -> NDArray[np.floating]:
    """Compute non-periodic pairwise distances using JAX.

    JAX auto-selects the best available backend (GPU > TPU > CPU).
    Deliberately does not use ``@clean_jax_cache`` -- see
    :func:`mdpp.analysis._backends._rmsd_matrix.rmsd_jax` for the
    rationale (clearing JAX's JIT compilation cache forces a slow
    recompile on every call and does not actually release device
    memory).

    Args:
        traj: Input trajectory.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.
        periodic: Ignored. Accepted for uniformity with
            :func:`distances_mdtraj`; this kernel does not support
            periodic boundary conditions.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` in float32 (jax
        inherits the float32 of ``traj.xyz``).  The wrapper casts
        ``copy=False`` to the user-resolved dtype.

    Raises:
        ImportError: If JAX is not installed.
        ValueError: If any pair index is out of range.
    """
    _validate_pairs(traj.n_atoms, pairs)
    _jax, jnp = require_jax()

    xyz_j = jnp.asarray(traj.xyz)
    pairs_j = jnp.asarray(pairs)
    diffs = xyz_j[:, pairs_j[:, 0], :] - xyz_j[:, pairs_j[:, 1], :]
    distances = jnp.sqrt(jnp.sum(diffs * diffs, axis=2))
    return np.asarray(distances)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

distance_backends: BackendRegistry[DistanceBackendFn] = BackendRegistry(default="mdtraj")
distance_backends.register("mdtraj", distances_mdtraj)
distance_backends.register("numba", distances_numba)
distance_backends.register("cupy", distances_cupy)
distance_backends.register("torch", distances_torch)
distance_backends.register("jax", distances_jax)
