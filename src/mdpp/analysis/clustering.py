"""Conformational clustering from RMSD matrices."""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from mdpp._dtype import resolve_dtype
from mdpp._types import DtypeArg
from mdpp.analysis._backends import RMSDBackend
from mdpp.analysis._backends._rmsd_matrix import rmsd_matrix_backends
from mdpp.core.trajectory import select_atom_indices

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_rmsd_matrix(
    traj: md.Trajectory,
    *,
    atom_selection: str = "backbone",
    backend: RMSDBackend = "mdtraj",
    dtype: DtypeArg = None,
) -> RMSDMatrixResult:
    """Compute an all-vs-all RMSD matrix between trajectory frames.

    Args:
        traj: Input trajectory.
        atom_selection: Atoms used for RMSD calculation.
        backend: Computation backend.  Defaults to ``"mdtraj"`` for
            API consistency with other analysis functions; switch to a
            faster backend explicitly when performance matters.

            - ``"mdtraj"`` (default) -- mdtraj precentered RMSD loop (CPU).
            - ``"numba"`` -- Numba-parallel QCP kernel (CPU, 50-200x faster).
            - ``"torch"`` -- PyTorch einsum + QCP (CUDA/CPU, float32).
            - ``"jax"`` -- JAX einsum + QCP (GPU/TPU/CPU, float32).
            - ``"cupy"`` -- CuPy einsum + QCP (CUDA, float32).

        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        RMSDMatrixResult with a symmetric ``(n_frames, n_frames)`` matrix.

    Raises:
        ValueError: If an unsupported backend is specified.
        ImportError: If the requested backend package is not installed.

    Memory note:
        Every backend returns its native ``float32`` output matrix
        (the numba kernel uses float64 accumulators internally but
        stores float32 in the result buffer; GPU kernels compute in
        float32 end-to-end).  This wrapper casts with ``copy=False``
        so when the resolved dtype is float32 (the package default)
        there is **no second copy** of the ``(n_frames, n_frames)``
        matrix.  For a 120k-frame trajectory this saves ~115 GB of
        peak RAM versus the old "cast to float64 for the Protocol
        contract, then cast back" path.  Passing ``dtype=np.float64``
        still forces a one-time upcast.
    """
    resolved = resolve_dtype(dtype)
    atom_indices = select_atom_indices(traj.topology, atom_selection)
    compute_fn = rmsd_matrix_backends.get(backend)
    rmsd_matrix = compute_fn(traj, atom_indices)

    # ``copy=False`` is critical at large N: when the backend's native
    # dtype already matches ``resolved`` we reuse the same buffer
    # instead of allocating a second ~N^2 matrix.
    return RMSDMatrixResult(
        rmsd_matrix_nm=rmsd_matrix.astype(resolved, copy=False),
        atom_indices=atom_indices,
    )


@njit(parallel=True, cache=True)
def _gromos_initial_counts(
    rmsd_matrix: NDArray[np.floating],
    cutoff_nm: float,
) -> NDArray[np.int64]:  # pragma: no cover - JIT-compiled
    """Count each row's neighbors within ``cutoff_nm`` (inclusive).

    Parallel over rows.  This runs once at the start of the GROMOS
    loop and dominates wall time for small cluster counts.
    """
    n = rmsd_matrix.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        c = 0
        for j in range(n):
            if rmsd_matrix[i, j] <= cutoff_nm:
                c += 1
        counts[i] = c
    return counts


@njit(cache=True)
def _gromos_loop(
    rmsd_matrix: NDArray[np.floating],
    counts: NDArray[np.int64],
    cutoff_nm: float,
) -> tuple[NDArray[np.int64], int, NDArray[np.int64]]:  # pragma: no cover - JIT-compiled
    """Run the incremental GROMOS greedy assignment loop.

    On entry ``counts[i]`` is the number of rows within ``cutoff_nm``
    of row ``i`` (including ``i`` itself).  The loop repeatedly picks
    the unassigned row with the largest count as the next cluster
    centre, assigns every unassigned neighbour of that centre, and
    **incrementally** decrements ``counts`` so that ``counts[i]`` is
    always the number of *still-unassigned* neighbours of ``i``.

    The incremental update is the key speed win: for each newly
    assigned member ``m``, we scan its row once (the matrix is
    symmetric, so its column is its row) and decrement ``counts`` at
    every neighbour.  This is ``O(|members| * n)`` per cluster
    instead of ``O(n^2)``.  For 120k frames with ~100 clusters the
    whole loop runs in ~10 seconds on a modern CPU, versus ~100
    minutes for the original fully-recomputing Python implementation.

    Returns:
        ``(labels, n_clusters, medoids)`` as three numpy arrays.
    """
    n = rmsd_matrix.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=np.bool_)
    # Pre-allocate medoid storage -- at most ``n`` clusters.
    medoids_buf = np.empty(n, dtype=np.int64)
    cluster_id = 0

    # Working buffer for newly-assigned member indices in the current
    # cluster -- sized for the worst case (first cluster might absorb
    # the entire trajectory).
    members_buf = np.empty(n, dtype=np.int64)

    while True:
        # Pick the unassigned row with the largest neighbour count.
        best = -1
        best_count = -1
        for i in range(n):
            if not assigned[i] and counts[i] > best_count:
                best_count = counts[i]
                best = i
        if best < 0:
            break

        # Collect this cluster's members: every unassigned row within
        # cutoff of ``best`` (includes ``best`` itself).
        n_members = 0
        for j in range(n):
            if not assigned[j] and rmsd_matrix[best, j] <= cutoff_nm:
                members_buf[n_members] = j
                n_members += 1
                labels[j] = cluster_id
                assigned[j] = True

        # Incrementally refresh ``counts``: for every newly assigned
        # member ``m``, each of ``m``'s neighbours loses one
        # still-unassigned neighbour.  Walk row ``m`` (cache-friendly
        # contiguous access) and decrement in place.
        for k in range(n_members):
            m = members_buf[k]
            for i in range(n):
                if rmsd_matrix[m, i] <= cutoff_nm:
                    counts[i] -= 1

        medoids_buf[cluster_id] = best
        cluster_id += 1

    return labels, cluster_id, medoids_buf[:cluster_id].copy()


def cluster_conformations(
    rmsd_matrix: NDArray[np.floating],
    *,
    method: str = "gromos",
    cutoff_nm: float = 0.15,
) -> ClusteringResult:
    """Cluster trajectory frames from a pairwise RMSD matrix.

    Uses a Numba-JIT'd GROMOS implementation that keeps **incremental**
    neighbour counts, so the inner loop is ``O(|cluster| * n)`` per
    cluster instead of the ``O(n^2)`` full recompute the original
    pure-Python version did.  For a 120k-frame trajectory with ~100
    clusters the runtime drops from impractical (hours) to ~10
    seconds.

    Args:
        rmsd_matrix: Symmetric pairwise RMSD matrix of shape ``(n, n)``
            in nm.  Accepts any floating dtype -- float32 is preferred
            at large ``n`` because the matrix itself is already the
            biggest allocation in the pipeline (57 GB at n=120k).
        method: Clustering method. ``"gromos"`` uses the GROMOS algorithm
            (largest-cluster-first greedy assignment).
        cutoff_nm: RMSD cutoff in nm for the GROMOS algorithm.

    Returns:
        ClusteringResult with per-frame labels and medoid frame indices.

    Raises:
        ValueError: If an unsupported method is specified.

    Memory note:
        No ``(n, n)`` auxiliary allocations.  Only ``O(n)`` working
        buffers (labels, counts, assigned, members_buf, medoids_buf).
        The ``rmsd_matrix`` itself is read-only.
    """
    if method != "gromos":
        raise ValueError(f"Unsupported clustering method: {method!r}. Use 'gromos'.")

    # ``np.ascontiguousarray`` is a no-op for already-contiguous inputs
    # (which is what ``compute_rmsd_matrix`` returns) and avoids a
    # numba strided-access slow path if the caller passed a view.
    rmsd_matrix = np.ascontiguousarray(rmsd_matrix)

    counts = _gromos_initial_counts(rmsd_matrix, cutoff_nm)
    labels, n_clusters, medoids = _gromos_loop(rmsd_matrix, counts, cutoff_nm)

    return ClusteringResult(
        labels=labels.astype(np.int_, copy=False),
        n_clusters=int(n_clusters),
        medoid_frames=medoids.astype(np.int_, copy=False),
    )
