"""Pairwise distance computation backends.

Provides four raw-array backends for non-periodic pairwise distances:

- ``numba`` -- Numba-parallel kernel on CPU.
- ``cupy`` -- CuPy vectorised operations on GPU.
- ``torch`` -- PyTorch vectorised operations on GPU/CPU.
- ``jax`` -- JAX/XLA vectorised operations on GPU/CPU.

The mdtraj backend is not included here because it has a different
call signature (takes a Trajectory object and supports periodic boundary
conditions).  It is handled directly by ``distance.py``.

All backends accept ``(xyz, pairs)`` and return a float64 numpy array
of shape ``(n_frames, n_pairs)``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from mdpp.analysis._backends._imports import require_cupy, require_jax, require_torch
from mdpp.analysis._backends._registry import BackendRegistry


def _validate_pairs(n_atoms: int, pairs: NDArray[np.int_]) -> None:
    """Raise ValueError if any pair index is out of range."""
    if pairs.size > 0 and (np.any(pairs < 0) or np.any(pairs >= n_atoms)):
        raise ValueError(
            f"atom_pairs must contain indices in [0, {n_atoms}), "
            f"got range [{int(pairs.min())}, {int(pairs.max())}]."
        )


def distances_numba(
    xyz: NDArray[np.float32],
    pairs: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute non-periodic pairwise distances using a Numba-parallel kernel.

    Parallelises the frame loop using ``prange``, giving ~5x speedup over
    mdtraj's single-threaded C/SSE kernel on multi-core machines.

    Args:
        xyz: Coordinates of shape ``(n_frames, n_atoms, 3)``.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` (float64).

    Raises:
        ValueError: If any pair index is out of range.
    """
    _validate_pairs(xyz.shape[1], pairs)

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


def distances_cupy(
    xyz: NDArray[np.float32],
    pairs: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute non-periodic pairwise distances on GPU using CuPy.

    Args:
        xyz: Coordinates of shape ``(n_frames, n_atoms, 3)``.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` (float64).

    Raises:
        ImportError: If CuPy is not installed.
        ValueError: If any pair index is out of range.
    """
    _validate_pairs(xyz.shape[1], pairs)
    cp = require_cupy()

    xyz_gpu = cp.asarray(xyz)
    pairs_gpu = cp.asarray(pairs)
    diffs = xyz_gpu[:, pairs_gpu[:, 0], :] - xyz_gpu[:, pairs_gpu[:, 1], :]
    distances = cp.sqrt(cp.sum(diffs * diffs, axis=2))
    return cp.asnumpy(distances).astype(np.float64)


def distances_torch(
    xyz: NDArray[np.float32],
    pairs: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute non-periodic pairwise distances using PyTorch.

    Uses CUDA if available, otherwise falls back to CPU.

    Args:
        xyz: Coordinates of shape ``(n_frames, n_atoms, 3)``.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` (float64).

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If any pair index is out of range.
    """
    _validate_pairs(xyz.shape[1], pairs)
    torch = require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        xyz_t = torch.as_tensor(np.ascontiguousarray(xyz), device=device)
        pairs_t = torch.as_tensor(pairs.astype(np.int64), device=device)
        diffs = xyz_t[:, pairs_t[:, 0], :] - xyz_t[:, pairs_t[:, 1], :]
        distances = torch.sqrt(torch.sum(diffs * diffs, dim=2))
    return distances.cpu().numpy().astype(np.float64)


def distances_jax(
    xyz: NDArray[np.float32],
    pairs: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute non-periodic pairwise distances using JAX.

    JAX auto-selects the best available backend (GPU > TPU > CPU).

    Args:
        xyz: Coordinates of shape ``(n_frames, n_atoms, 3)``.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` (float64).

    Raises:
        ImportError: If JAX is not installed.
        ValueError: If any pair index is out of range.
    """
    _validate_pairs(xyz.shape[1], pairs)
    _jax, jnp = require_jax()

    xyz_j = jnp.asarray(xyz)
    pairs_j = jnp.asarray(pairs)
    diffs = xyz_j[:, pairs_j[:, 0], :] - xyz_j[:, pairs_j[:, 1], :]
    distances = jnp.sqrt(jnp.sum(diffs * diffs, axis=2))
    return np.asarray(distances).astype(np.float64)


# ---------------------------------------------------------------------------
# Registry (raw-array backends only; mdtraj handled by distance.py)
# ---------------------------------------------------------------------------

distance_backends: BackendRegistry = BackendRegistry(default="numba")
distance_backends.register("numba", distances_numba)
distance_backends.register("cupy", distances_cupy)
distance_backends.register("torch", distances_torch)
distance_backends.register("jax", distances_jax)
